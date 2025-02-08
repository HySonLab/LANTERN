import torch
#from sklearn import metrics
import numpy as np
import pickle,os
#from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from util_representations import load_relation_embed, load_entity_embed, get_bio_bert
from collections import defaultdict
import json
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def sigmoid(x):
    """Apply sigmoid function to x."""
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def calc_auc(labels, scores):
    auc = roc_auc_score(labels, scores)
    return auc

def calc_aupr(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)
    aupr = auc_precision_recall
    return aupr

def calc_other_metrics(y_true, y_pred):
    """
    Calculate accuracy, sensitivity (recall), specificity, precision, F1 score, and MCC.
    
    Args:
        y_true (list or np.array): Ground truth labels (0 or 1).
        y_pred (list or np.array): Predicted labels (0 or 1).
    
    Returns:
        dict: A dictionary with calculated metrics.
    """
    # Confusion matrix
    probabilities = sigmoid(y_pred)
    
    # Convert probabilities to binary predictions with a threshold of 0.5
    y_pred = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Return metrics as a dictionary
    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "mcc": mcc
    }
    
    return metrics

def save_model(result, path, mode='ab'):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, mode) as f:
            pickle.dump(result, f)
        print(f"Model saved successfully at {path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

def load_model(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


def print_config(config, args):
    print('data_path:', args.data_path)
    print('learning rate:', args.lr)
    print('embed_size:', args.embed_dim)
    print('weight_decay:', args.decay)
    print('dropout:', args.dropout)
    print('neg_ratio:', args.neg_ratio)
    print('enable_augmentation:', args.enable_augmentation)
    print('enable_gumbel:', args.enable_gumbel)
    print('gumbel tau:', args.tau)
    print('gumbel amplitude:', args.amplitude)
    print('valid_step', args.valid_step)

    for k, v in config.items():
        string = '{}:{}'.format(k, v)
        print(string)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def exchange_head_tail(triple):
    h, r, t = triple
    return t, r, h


def generate_directed_coo_matrix(triples, dd_num):
    row = []
    data = []
    col = []
    dd_num = dd_num - 1
    # Why so many cases like this ? Remember, when hop == 1, other data is 
    for triple in triples:
        h, r, t = triple
        if h <= dd_num and t <= dd_num: 
            row.append(h)
            data.append(r)
            col.append(t)
            row.append(t)
            data.append(r)
            col.append(h)
        elif h > dd_num and t > dd_num:
            print('check dataset')
        elif h <= dd_num < t:
            h, r, t = exchange_head_tail(triple)
            row.append(h)
            col.append(t)
            data.append(r)
        elif h > dd_num >= t:
            row.append(h)
            col.append(t)
            data.append(r)
    return (row, col), data


def exclude_isolation_point(train_set, valid_set):
    head_set = set()
    tail_set = set()
    exclusion_list = []
    valid = []
    valid_set = valid + valid_set
    for train_triple in train_set:
        h, _, t = train_triple
        head_set.add(h)
        tail_set.add(t)
    for valid_triple in valid_set:
        h, _, t = valid_triple
        if h not in head_set : # previously : and t not in tail_set:
            exclusion_list.append(valid_triple)
            valid_set.remove(valid_triple)
    return valid_set, exclusion_list


def calc_norm(x):
    x = x.to('cpu').numpy().astype('float32')
    x[x == 0.] = np.inf
    x = torch.FloatTensor(1. / np.sqrt(x))
    return x.unsqueeze(1)


def comp(list1, list2):
    for val in list1:
        if val in list2:
            return True
    return False

def get_top_ranked_drugs(probs, k) :
    """
    return top k drugs ranked by probabilities 
    """
    return torch.topk(probs, k).indices

def create_pyg_dataset(embed_dim, kg_coo, kg_relation, entity_embed_root = None, relation_embed_root = None) :
    entity_num = max(max(row) for row in kg_coo) + 1
    if entity_embed_root is None :
        x = torch.randn([entity_num, embed_dim])
    else :
        x = load_entity_embed(entity_num, entity_embed_root)
    #print(kg_coo.shape())
    num_edges = len(kg_coo[0]) # 2, num_edges
    print(251, 'utils, num_edges', num_edges)

    if relation_embed_root is None :
        edge_attr = torch.randn([num_edges, embed_dim])
    else :
        edge_attr = load_relation_embed(kg_relation, relation_embed_root) # num_edges, embed_dim
    
    data = Data(x, kg_coo, edge_attr)
    return data

def get_biobert_representation(input) :
    tokenizer, model = get_bio_bert()
    input = tokenizer([input], return_tensors="pt", truncation=True, padding=True, max_length=512)
  
    with torch.no_grad():
        outputs = model(**input)
        last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
        embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
    return embeddings

def random_initialize(shape):
    return torch.tensor(np.random.randn(*shape), dtype=torch.float)

#        return (torch.tensor(head), torch.tensor(tail), torch.tensor(positive_samples), torch.tensor(negative_samples),
#                torch.tensor(positive_labels, dtype=torch.float), torch.tensor(negative_labels, dtype=torch.float))
def custom_collate(batch):
    #batch = [item for item in batch if item is not None]
    try :
        
            #print(279, 'utils', batch)
            # Extracting individual components from the batch
            h_batch = [item[0] for item in batch if item is not None]
            t_batch = [item[1] for item in batch if item is not None]
            pos_t_batch = [item[2] for item in batch if item is not None]
            neg_t_batch = [item[3] for item in batch if item is not None]
            pos_labels_batch = [item[4] for item in batch if item is not None]
            neg_labels_batch = [item[5] for item in batch if item is not None]

            # Calculating the maximum lengths for padding
            max_len_pos_t = max(len(pos_t) for pos_t in pos_t_batch)
            max_len_n_t = max(len(neg_t) for neg_t in neg_t_batch)
            max_len_pos_labels = max(len(pos_label) for pos_label in pos_labels_batch)
            max_len_neg_labels = max(len(neg_label) for neg_label in neg_labels_batch)

            # Function to pad sequences and generate a mask
            def pad_and_mask(indices, max_len):
                padded_indices = torch.zeros((len(indices), max_len), dtype=torch.long)
                mask = torch.zeros((len(indices), max_len), dtype=torch.bool)
                for i, idx in enumerate(indices):
                    length = len(idx)
                    # Ensure idx is a tensor before padding
                    padded_indices[i, :length] = torch.tensor(idx).clone().detach() if not isinstance(idx, torch.Tensor) else idx
                    mask[i, :length] = torch.ones(length).clone().detach()
                return padded_indices, mask

            def pad_and_mask_pos_labels(indices, max_len):
                padded_indices = torch.ones((len(indices), max_len), dtype=torch.long)
                mask = torch.zeros((len(indices), max_len), dtype=torch.bool)
                for i, idx in enumerate(indices):
                    length = len(idx)
                    # Ensure idx is a tensor before padding
                    padded_indices[i, :length] = torch.tensor(idx).clone().detach() if not isinstance(idx, torch.Tensor) else idx
                    mask[i, :length] = 1
                return padded_indices, mask

            # Padding and masking for positive and negative triples and labels
            pos_t_padded, pos_t_mask = pad_and_mask(pos_t_batch, max_len_pos_t)
            neg_t_padded, neg_t_mask = pad_and_mask(neg_t_batch, max_len_n_t)
            pos_labels_padded, pos_labels_mask = pad_and_mask_pos_labels(pos_labels_batch, max_len_pos_labels)
            neg_labels_padded, neg_labels_mask = pad_and_mask(neg_labels_batch, max_len_neg_labels)

            # Creating a mask dictionary
            mask = defaultdict(list)
            mask['pos_t'] = pos_t_mask.clone().detach()
            mask['neg_t'] = neg_t_mask.clone().detach()
            mask['pos_labels'] = pos_labels_mask.clone().detach()
            mask['neg_labels'] = neg_labels_mask.clone().detach()

            # Convert h_batch and t_batch into tensors if not already
            h_batch_tensor = torch.tensor(h_batch) if not isinstance(h_batch, torch.Tensor) else h_batch.clone().detach()
            t_batch_tensor = torch.tensor(t_batch) if not isinstance(t_batch, torch.Tensor) else t_batch.clone().detach()
            #n_t_batch_tensor = torch.tensor(t_batch) if not isinstance(t_batch, torch.Tensor) else t_batch.clone().detach()
            #print(330,'utils pos t padded and sum of mask', pos_t_padded.shape, pos_t_mask.sum())
            return h_batch_tensor, t_batch_tensor, pos_t_padded.clone().detach(), neg_t_padded.clone().detach(), pos_labels_padded.clone().detach(),neg_labels_padded.clone().detach(), mask
            #return h_batch_tensor, t_batch_tensor, pos_t_padded.clone().detach(), n_t_batch_tensor, pos_labels_padded.clone().detach(), mask
    except :
        #print(322, 'utils', batch)
        return None


def get_smiles_db() :
    import json
    import pandas as pd

    # Load drugbank_db.json
    with open('drugbank_db/drugbank_db.json', 'r') as f:
        drugbank_db = json.load(f)

    # Extract DrugBank IDs
    
    drugbank_ids = list(drugbank_db.keys())
    print(len(drugbank_ids))
    # Load drugbank.csv
    drugbank_df = pd.read_csv('drugbank_db/drugbank.csv')

    # Create a mapping of DrugBank ID to SMILES
    smiles_mapping = {}
    for index, row in drugbank_df.iterrows():
        drug_id = row['Drug id'].strip()
        print(309, drug_id)
        smile = row['smiles']
        print(311, smile)
        if drug_id in drugbank_ids:
            print(313)
            smiles_mapping[drug_id] = smile
        else :
            print(316, drug_id, len(drug_id))
    print(len(list(smiles_mapping.values())))
    # Save the mapping to drugsmiles_db.json
    with open('drugbank_db/drugsmiles_db.json', 'w') as f:
        json.dump(smiles_mapping, f, indent=4)

    print("drugsmiles_db.json created successfully.")

def get_molformer_representation():

    # Set visible devices and use cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load drugbank_db.json
    with open('drugbank_db/drugsmiles_db.json', 'r') as f:
        drugbank_smiles_db = json.load(f)

    # Load model and tokenizer
    from util_representations import get_molformer
    tokenizer, model = get_molformer()

    # Tokenize the SMILES strings
    inputs = tokenizer(list(drugbank_smiles_db.values()), padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Move the model to GPU
    model = model.to(device)

    # Number of SMILES sequences, not the number of keys
    num_inputs = inputs['input_ids'].shape[0]  
    batch_size = 30
    num_batch = num_inputs // batch_size + int(num_inputs % batch_size != 0)

    all_embeddings = []

    with torch.no_grad():
        for i in range(num_batch):
            # Slice batch inputs based on the number of sequences
            batch_inputs = {k: v[i * batch_size:min((i + 1) * batch_size, num_inputs)] for k, v in inputs.items()}
            outputs = model(**batch_inputs)
            
            # If model has pooler_output, otherwise check the output format
            last_hidden_states = outputs.pooler_output  
            
            # Average embeddings for this batch
            batch_embeddings = last_hidden_states
            
            print(f'{min((i+1)*batch_size, num_inputs)} elements passed through the model')
            all_embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    print(337, embeddings.shape)
    # Convert embeddings to Python list
    embeddings_list = embeddings.tolist()

    # Create the dictionary of DrugBank IDs to embeddings
    drugbank_ids = list(drugbank_smiles_db.keys())
    print(328, len(drugbank_ids), len(embeddings_list))
    drugsmiles_db_pretrained = {drugbank_ids[i]: embeddings_list[i] for i in range(len(drugbank_ids))}

    # Save the dictionary to a JSON file
    with open('drugsmiles_db_pretrained.json', 'w') as f:
        json.dump(drugsmiles_db_pretrained, f)

    print("drugsmiles_db_pretrained.json created successfully.")


def get_biobert_representation_batches():

    # Load drugbank_db.json , drug description
    with open('drugbank_des_db_updated.json', 'r') as f:
        drugbank_db = json.load(f)

    # Load model and tokenizer
    from util_representations import get_bio_bert
    tokenizer, model = get_bio_bert()

    # Tokenize the SMILES strings
    inputs = tokenizer(list(drugbank_db.values()), return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Move the model to GPU
    model = model.to(device)

    # Number of SMILES sequences, not the number of keys
    num_inputs = inputs['input_ids'].shape[0]  
    batch_size = 30
    num_batch = num_inputs // batch_size + int(num_inputs % batch_size != 0)

    all_embeddings = []

    with torch.no_grad():
        for i in range(num_batch):
            # Slice batch inputs based on the number of sequences
            batch_inputs = {k: v[i * batch_size:min((i + 1) * batch_size, num_inputs)] for k, v in inputs.items()}
            outputs = model(**batch_inputs)
            
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
            batch_embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
            print(f'{min((i+1)*batch_size, num_inputs)} elements passed through the model')
            all_embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    print(337, embeddings.shape)
    # Convert embeddings to Python list
    embeddings_list = embeddings.tolist()

    # Create the dictionary of DrugBank IDs to embeddings
    drugbank_ids = list(drugbank_db.keys())
    print(328, len(drugbank_ids), len(embeddings_list))
    drugsmiles_db_pretrained = {drugbank_ids[i]: embeddings_list[i] for i in range(len(drugbank_ids))}

    # Save the dictionary to a JSON file
    with open('drugdes_db_pretrained.json', 'w') as f:
        json.dump(drugsmiles_db_pretrained, f)

    print("drugdes_db_pretrained.json created successfully.")

def normalize_embedding(embedding):
    """Normalize an embedding using L2 norm."""
    norm = torch.norm(embedding, p=2)
    return embedding / norm if norm != 0 else embedding

def find_most_similar_drug(embed_x, embed):
    """Find the most similar drug to embed_x using PyTorch."""
    # Normalize the embedding for drug x
    embed_x_normalized = normalize_embedding(embed_x)

    # Prepare a list of drug IDs and their corresponding embeddings as PyTorch tensors
    drug_ids = list(embed.keys())
    drug_embeddings = torch.stack([normalize_embedding(embed[drug_id]) for drug_id in drug_ids])

    # Compute the dot products between the normalized drug embeddings and embed_x
    similarities = torch.matmul(drug_embeddings, embed_x_normalized)

    # Find the index of the highest similarity
    most_similar_idx = torch.argmax(similarities) # the position that has highest similarity
    most_similar_id = drug_ids[most_similar_idx]  # the most similar drug ids 
    highest_similarity = similarities[most_similar_idx].item()

    return most_similar_id, highest_similarity

def custom_binary_cross_entropy_with_logits(score, label, pos_weight) :
    z = torch.sigmoid(score)
    print(500, 'utils', score, label, -(pos_weight*label*torch.log(z) + (1-label)*torch.log(1-z)))
    return -(pos_weight*label*torch.log(z) + (1-label)*torch.log(1-z))
"""
if __name__=='__main__':
    #get_biobert_representation_batches()
    #get_smiles_db()
    #get_molformer_representation()
"""    
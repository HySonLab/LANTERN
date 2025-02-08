import pickle
import torch
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import re


def get_bio_bert():
    # Load BioBERT model and tokenizer
    model_name = "dmis-lab/biobert-v1.1"  # Use the BioBERT model from Hugging Face
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model

def get_protbert() :
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
    return tokenizer, model  
  
def get_molformer() :
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    return tokenizer, model

    
def load_entity_embed(entity_num, entity_embed_root) :
    # Load relation embeddings for graph
    try :
        with open (entity_embed_root, 'rb') as f :
            res = pickle.load(f)
            return res[:entity_num]
    except :
        print("Error ! Loading relation embedding failed")


def load_relation_embed(relation, relation_embed_root) :
    # Load relation embeddings for graph
    try :
        with open(relation_embed_root, 'rb') as f :
            res = pickle.load(f)
            return res[relation]
    except :
        print("Error ! Loading relation embedding failed")


def get_geneid2seq(csv_paths, column_key_name = 'Gene', column_value_name='Target Sequence') :
    """
    Input : 
    csv_paths : train, val, test
    column_name : Name of coloumn to collect
    Output :
    Unique gene id to coloumn data
    """
    gene_sequence_dict = {}
    for csv_path in csv_paths :
        df = pd.read_csv(csv_path)
        unique_genes = df[column_key_name].unique()
        try :
            print(len(unique_genes))
        except :
            print("Error when trying to print the len of unique genes")
        for gene in unique_genes :
            if gene not in gene_sequence_dict.keys() : 
                sequence = df.loc[df[column_key_name] == gene][column_value_name].unique()[0] # 0 : Series to number
                gene_sequence_dict[gene] = sequence
    return gene_sequence_dict

def get_protbert_embeddings(sequence) :
    tokenizer, model = get_protbert() # get protein sequence representation model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    inputs = tokenizer([sequence], do_lower_case=False )
    print(inputs) # A dict, keys () : input_ids, ...;
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_size = 30
    num_inputs = inputs['input_ids'].shape[0]
    num_batch = num_inputs//batch_size + int(num_inputs % batch_size != 0)
    all_embeddings = []
    with torch.no_grad():
        for i in range(num_batch) :
            batch_inputs = {k : v[(i * batch_size):min((i+1)*batch_size, num_inputs)] for k, v in inputs.items()}
            batch_inputs = re.sub(r"[UZOB]", "X", batch_inputs)
            outputs = model(**batch_inputs)
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
            batch_embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
            print(f'{min((i+1)*batch_size, num_inputs)} elements passed through the model')
            all_embeddings.append(batch_embeddings)
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings.cpu().numpy()

def get_bio_bert_embeddings(description) :
    tokenizer, model = get_bio_bert()
    inputs = tokenizer([description], return_tensors="pt", truncation=True, padding=True, max_length=512)
    print(inputs)
    print(type(inputs))
 
    batch_size = 15
    num_inputs = inputs['input_ids'].shape[0]
    num_batch = num_inputs//batch_size + int(num_inputs % batch_size != 0)
    print(num_inputs, num_batch)
    
    all_embeddings = []
    with torch.no_grad():
        for i in range(num_batch) :
            batch_inputs = {k : v for k, v in inputs.items()}
            outputs = model(**batch_inputs)
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
            batch_embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
            print(f'{min((i+1)*batch_size, num_inputs)} elements passed through the model')
            all_embeddings.append(batch_embeddings)
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings.cpu().numpy()
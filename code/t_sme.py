import torch
from torch import nn
from torch.nn import functional as fn

import pickle

import pandas as pd
import numpy as np
import json

import random 
import os
import sys
import io

import csv
import ast  # For safely evaluating the string as a Python list

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
code_folder = os.path.dirname(__file__)
kgcnh_folder = os.path.dirname(code_folder)
kgdrp_folder = os.path.dirname(kgcnh_folder)

kgcnh_path = os.path.dirname(os.path.dirname(__file__))
biosnap_path = os.path.join(kgcnh_path, 'data', 'BioSNAP')

model_save_path = os.path.join(kgcnh_folder, 'log','training','BioSNAP', '1', 'result_0.992936974798785_19.pkl')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
 
class Custom_CPU_Unpickler(pickle.Unpickler):     
    def find_class(self, module, name):
        if module == "model":
            module = "KGCNH.code.model"
        elif module == "layers":
            module = "KGCNH.code.layers"
        # Special case for torch.storage._load_from_bytes
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

    
def load_cpu_pickle_model() :
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    seed = 24 # 120 auc : 0.963, aupr : 0.949; 10 auc : 0.955, aupr : 0.937; 1200 : 0.953, 0.943, result_aupr = 0.99135, num_head = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(model_save_path, "rb") as f:
      res = Custom_CPU_Unpickler(f).load()
      #model = torch.load(f, map_location=torch.device('cpu'))
    print("Type res :", type(res))
    print("Keys of res :" , res.keys())
    model = res['model']
    #model.dataset_name = 'DAVIS'
    #model.path_2_pretrained_embedding = os.path.join(os.path.dirname(kgcnh_folder), 'embeddings')
    #model.modality = 1
    model.eval()
    return model

def create_pretrained_embedding_csv(output_csv_path, test_csv_path, drug_embed_json_path, gene_embed_json_path):
    # Step 1: Load the test.csv file
    test_df = pd.read_csv(test_csv_path)

    # Step 2: Load the DrugBank and Gene embeddings from JSON files
    with open(drug_embed_json_path, 'r') as f:
        drug_embeddings = json.load(f)
    with open(gene_embed_json_path, 'r') as f:
        gene_embeddings = json.load(f)

    # Step 3: Create a list to store rows for the final DataFrame
    data = []

    model = load_cpu_pickle_model()
    count = 0
    # Step 4: Process each row in the test_df
    for _, row in test_df.iterrows():
        drug_id = row['DrugBank ID']
        gene_id = row['Gene']
        label = row['Label']

        # Get embeddings for DrugBank ID and Gene
        drug_embed = torch.tensor(drug_embeddings.get(drug_id), dtype=torch.float)
        gene_embed = torch.tensor(gene_embeddings.get(gene_id), dtype=torch.float)

        drug_embed = model.drug_smiles_project(drug_embed)
        gene_embed = model.gene_sequence_project(gene_embed)


        # Check if embeddings exist
        if drug_embed is not None and gene_embed is not None:
            # Concatenate the embeddings
            embed = torch.concat([drug_embed, gene_embed], dim = -1)
            if count % 1000 == 0 :
                print(embed.shape)
            # Add to data list
            data.append({'embed': embed.tolist(), 'label': label})
        else:
            print(f"Warning: Missing embedding for DrugBank ID: {drug_id} or Gene: {gene_id}")
        count = count + 1
    # Step 5: Create the final DataFrame and save it to a CSV file
    output_df = pd.DataFrame(data)
    output_df.to_csv(output_csv_path, index=False)

    print(f"CSV file created at: {output_csv_path}")


def get_embed_label_from_csv(csv_file_path) :
    embeddings = []
    labels = []

    with open(csv_file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            embed = np.array(ast.literal_eval(row["embedding"]))  # Safely evaluate the string
            label = int(row["label"])
            embeddings.append(embed)
            labels.append(label)
    return embeddings, label

# transformer embedding of entities
def create_trained_embedding_csv(output_csv_path, test_csv_path, drug_embed_json_path, gene_embed_json_path) :
    # Step 1: Load the test.csv file
    test_df = pd.read_csv(test_csv_path)

    # Step 2: Load the DrugBank and Gene embeddings from JSON files
    with open(drug_embed_json_path, 'r') as f:
        drug_embeddings = json.load(f)
    with open(gene_embed_json_path, 'r') as f:
        gene_embeddings = json.load(f)

    # Step 3: Create a list to store rows for the final DataFrame
    data = []

    model = load_cpu_pickle_model()
    count = 0
    # Step 4: Process each row in the test_df
    for _, row in test_df.iterrows():
        drug_id = row['DrugBank ID']
        gene_id = row['Gene']
        label = row['Label']
        
        #print(150, drug_id, gene_id)
        if drug_id in model.train_entity2index :
            drug_index = model.train_entity2index[drug_id]
            drug_embed = model.entity_embed[drug_index]
        else :
            drug_embed = torch.tensor(drug_embeddings.get(drug_id), dtype=torch.float)
            drug_embed = model.drug_smiles_project(drug_embed)
        if gene_id in model.train_entity2index :
            gene_index = model.train_entity2index[gene_id]
            gene_embed = model.entity_embed[gene_index]
        else :
            gene_embed = torch.tensor(gene_embeddings.get(gene_id), dtype=torch.float)
            gene_embed = model.gene_sequence_project(gene_embed)


        # Check if embeddings exist
        if drug_embed is not None and gene_embed is not None:
            # Concatenate the embeddings
            embed = torch.concat([drug_embed, gene_embed], dim = -1)
            embed = fn.relu(model.dropout(model.predictor1[0](embed.unsqueeze(0)))).squeeze(0)
            if count % 1000 == 0 :
                print(embed.shape)
            # Add to data list
            data.append({'embed': embed.tolist(), 'label': label})
        else:
            print(f"Warning: Missing embedding for DrugBank ID: {drug_id} or Gene: {gene_id}")
        count = count + 1
    # Step 5: Create the final DataFrame and save it to a CSV file
    output_df = pd.DataFrame(data)
    output_df.to_csv(output_csv_path, index=False)

    print(f"CSV file created at: {output_csv_path}")

def visualize_tsne(csv_path):
    # Step 1: Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert the 'embed' column from string to numpy arrays
    df['embed'] = df['embed'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

    # Step 2: Separate the data into features (embeddings) and labels
    X = np.vstack(df['embed'].values)  # Stack arrays into a 2D numpy array
    y = df['label'].values            # Labels (assumes 'label' column exists)

    # Step 3: Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42, learning_rate=50)
    X_embedded = tsne.fit_transform(X)

    # Step 4: Plot the 2D representation
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']  # Add more if needed
    
    for idx, label in enumerate(unique_labels):
        plt.scatter(
            X_embedded[y == label, 0], 
            X_embedded[y == label, 1], 
            label=f'Label {label}', 
            alpha=0.6, 
            c=colors[idx % len(colors)]
        )
    
    plt.title('t-SNE Visualization of BioSNAP pair embedding after Transformer')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ =='__main__' :
    # Example usage
    """
    create_pretrained_embedding_csv(
        output_csv_path='E:/TMI/KGCNH/data/BioSNAP/t_sme.csv',
        test_csv_path='E:/TMI/KGCNH/data/BioSNAP/test.csv',
        drug_embed_json_path='E:/TMI/embeddings/BioSNAP/pretrained/drug/drug_smiles.json',
        gene_embed_json_path='E:/TMI/embeddings/BioSNAP/pretrained/gene/gene_sequence.json'
    )
    """

    #model = load_cpu_pickle_model()
    #print(len(model.train_entity2index))
    #print(type(model))
    #t_sme_pretrained_path = 'E:/TMI/KGCNH/data/BioSNAP/t_sme.csv'
    #visualize_tsne(t_sme_pretrained_path)

    """
    create_trained_embedding_csv(
        output_csv_path='E:/TMI/KGCNH/data/BioSNAP/t_sme_transformer.csv',
        test_csv_path='E:/TMI/KGCNH/data/BioSNAP/test.csv',
        drug_embed_json_path='E:/TMI/embeddings/BioSNAP/pretrained/drug/drug_smiles.json',
        gene_embed_json_path='E:/TMI/embeddings/BioSNAP/pretrained/gene/gene_sequence.json'
    )
    """

    t_sme_transformer_path = 'E:/TMI/KGCNH/data/BioSNAP/t_sme_transformer.csv'
    visualize_tsne(t_sme_transformer_path)
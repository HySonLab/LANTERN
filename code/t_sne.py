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
    #visualize_tsne()
    return
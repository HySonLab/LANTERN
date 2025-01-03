import argparse
import torch
import os
import pickle
import sys
import re
from util_representations import load_relation_embed, load_entity_embed, get_bio_bert
from transformers import BertModel, BertTokenizer
import re
import json
import time
import pandas as pd
import requests
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adjust sys.path to ensure modules are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
code_folder = os.path.dirname(__file__)
kgcnh_folder = os.path.dirname(code_folder)
kgdrp_folder = os.path.dirname(kgcnh_folder)

train_df_path = os.path.join(kgcnh_folder, 'data' , 'BioSNAP', 'train2.csv')
train_df = pd.read_csv(train_df_path)
valid_df_path = os.path.join(kgcnh_folder, 'data' , 'BioSNAP', 'val2.csv')
valid_df = pd.read_csv(valid_df_path)
test_df_path = os.path.join(kgcnh_folder, 'data' , 'BioSNAP', 'test2.csv')
test_df = pd.read_csv(test_df_path)

def save_gene_to_sequence_mapping(train_df, test_df, val_df, output_path):
    """
    Creates a JSON file mapping Protein UniProt IDs (from the 'Gene' column) to their sequences
    (from the 'Sequence' column) and saves it to 'gene2seq.json' in the specified output_path.

    Args:
        train_df (pd.DataFrame): The training dataset with columns 'Gene' and 'Sequence'.
        test_df (pd.DataFrame): The testing dataset with columns 'Gene' and 'Sequence'.
        val_df (pd.DataFrame): The validation dataset with columns 'Gene' and 'Sequence'.
        output_path (str): The directory path where 'gene2seq.json' will be saved.

    Returns:
        None
    """
    # Combine the three DataFrames
    combined_df = pd.concat([train_df, test_df, val_df])
    
    # Drop duplicates to ensure unique UniProt IDs
    unique_df = combined_df.drop_duplicates(subset='Gene', keep='first')
    
    # Create a dictionary mapping 'Gene' to 'Sequence'
    gene_to_seq_mapping = unique_df.set_index('Gene')['Target Sequence'].to_dict()
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the mapping to a JSON file
    output_file = os.path.join(output_path, 'gene2seq.json')
    with open(output_file, 'w') as json_file:
        json.dump(gene_to_seq_mapping, json_file, indent=4)
    
    print(f"'gene2seq.json' saved to {output_file}")

# Example Usage
#save_gene_to_sequence_mapping(train_df, test_df, valid_df, os.path.join(kgdrp_folder, 'protein_3d'))
def download_pdb_files(json_file_path, output_dir):
    """
    Downloads PDB files for proteins listed in a JSON file mapping UniProt IDs to sequences.
    If the PDB file is unavailable, logs the UniProt ID and counts the unfound proteins.

    Args:
        json_file_path (str): Path to the gene2seq.json file.
        output_dir (str): Directory where PDB files should be saved.
    
    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the UniProt IDs from the JSON file
    with open(json_file_path, 'r') as json_file:
        gene_to_seq = json.load(json_file)
    
    base_url = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"
    unfound_proteins = []

    # Iterate over UniProt IDs
    for uniprot_id in tqdm(gene_to_seq.keys(), desc="Downloading PDB files"):
        pdb_url = base_url.format(uniprot_id)
        output_file = os.path.join(output_dir, f"{uniprot_id}.pdb")
        
        try:
            # Attempt to download the PDB file
            response = requests.get(pdb_url, timeout=30)
            if response.status_code == 200:
                with open(output_file, 'wb') as pdb_file:
                    pdb_file.write(response.content)
            else:
                print(f"PDB file not found for UniProt ID: {uniprot_id}")
                unfound_proteins.append(uniprot_id)
        except requests.RequestException as e:
            print(f"Error fetching PDB file for UniProt ID: {uniprot_id}, Error: {e}")
            unfound_proteins.append(uniprot_id)
        

    # Log unfound proteins
    if unfound_proteins:
        unfound_count = len(unfound_proteins)
        print(f"\n{unfound_count} PDB files could not be found.")
        
        unfound_log_path = os.path.join(output_dir, "unfound_proteins.log")
        with open(unfound_log_path, 'w') as log_file:
            for protein in unfound_proteins:
                log_file.write(f"{protein}\n")
        print(f"Unfound UniProt IDs logged to {unfound_log_path}")
        
    else:
        print("\nAll PDB files downloaded successfully!")

# Example Usage
download_pdb_files(os.path.join(kgdrp_folder, 'protein_3d', 'gene2seq.json'), os.path.join(kgdrp_folder, 'protein_3d'))
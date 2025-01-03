import pandas as pd
import requests
from util_representations import * # get bio bert , 
import os
import json
import torch
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Example usage:
csv_paths = [r'E:\Research\Hysonlab\KGDRP\KGCNH\data\BioSNAP\train.csv', 
             r'E:\Research\Hysonlab\KGDRP\KGCNH\data\BioSNAP\val.csv', 
             r'E:\Research\Hysonlab\KGDRP\KGCNH\data\BioSNAP\test.csv']
                  # Replace with your input CSV file path
output_csv_paths = [r'E:\Research\Hysonlab\KGDRP\KGCNH\data\BioSNAP\train2.csv', 
             r'E:\Research\Hysonlab\KGDRP\KGCNH\data\BioSNAP\val2.csv', 
             r'E:\Research\Hysonlab\KGDRP\KGCNH\data\BioSNAP\test2.csv']  # Replace with your desired output file path

def get_gene_function_description(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        # Extracting function description from comments
        if "comments" in data:
            for comment in data["comments"]:
                if comment.get("commentType") == "FUNCTION":
                    for text in comment.get("texts", []):
                        return text.get("value")
        return "Function description not available."
    else:
        return "Failed to retrieve data."

def add_gene_function_column(csv_paths, output_csv_paths):
    # Create a dictionary to store UniProt ID -> gene function mappings
    gene_function_map = {}
    for csv_path, output_csv_path in zip(csv_paths, output_csv_paths) :
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Extract unique UniProt IDs from the 'Gene' column
        unique_genes = df['Gene'].unique()

        # Retrieve gene functions for each unique UniProt ID
        count = 0
        for gene in unique_genes:
            if gene not in gene_function_map.keys() :
                description = get_gene_function_description(gene)
                gene_function_map[gene] = description
            else :
                print(gene, "in", csv_path, "already exists")
            count+=1
            if count % 100 == 0 and count > 0 :
                print('Retrieved 100 gene description')
        print("Retrieval done")     
        # Map the gene functions to the 'Gene' column
        df['gene_function'] = df['Gene'].map(gene_function_map)
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv_path, index=False)
    return gene_function_map


def get_bio_bert_embeddings(gene_function_map, save_embed_path) :
    tokenizer, model = get_bio_bert()
    inputs = tokenizer(list(gene_function_map.values()), return_tensors="pt", truncation=True, padding=True, max_length=512)
    print(inputs)
    print(type(inputs))
 
    batch_size = 15
    num_inputs = inputs['input_ids'].shape[0]
    num_batch = num_inputs//batch_size + int(num_inputs % batch_size != 0)
    print(num_inputs, num_batch)
    
    all_embeddings = []
    with torch.no_grad():
        for i in range(num_batch) :
            batch_inputs = {k : v[(i * batch_size):min((i+1)*batch_size, num_inputs)] for k, v in inputs.items()}
            outputs = model(**batch_inputs)
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
            batch_embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
            print(f'{min((i+1)*batch_size, num_inputs)} elements passed through the model')
            all_embeddings.append(batch_embeddings)
    embeddings = torch.cat(all_embeddings, dim=0)
    print('embeddings: ', embeddings.shape)
    print(embeddings[0][:])
    
    # Create a dictionary to store UniProtID -> BioBERT embedding
    try :
      first_gene_function = tokenizer(list(gene_function_map.values())[0], return_tensors="pt", truncation=True, padding=True, max_length=512)
      first_gene_function_embed = model(**first_gene_function)
      first_gene_function_embed = first_gene_function_embed.last_hidden_state  # (batch_size, seq_length, hidden_dim)
      first_gene_function_embed = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)      
      #print(89, first_gene_function_embed)
      #print(90, embeddings[0][:])
    except :
      print("Error, 94")
    
    # Save the resulting dictionary to the specified file path
    res = {uniprot_id: embeddings[i].tolist()
        for i, uniprot_id in enumerate(list(gene_function_map.keys()))}

    # Save the resulting dictionary to the specified file path
    try :
      with open(save_embed_path, 'w') as f:
          #print(106)
          json.dump(res, f, indent=4)
          #print(108)
      print(f"BioBERT embeddings saved to {save_embed_path}")
      return gene_function_map, embeddings
    except :
      print("Can not save embed to the path !")
      return gene_function_map, embeddings
    

def get_drug2SmiAndDes(csv_paths, output_json_paths, drug_description_path):
    drug_smiles_map = {}
    drug_df = pd.read_csv(drug_description_path)
    #print(drug_df.columns.tolist())
    # Create a dictionary to store UniProt ID -> gene function mappings
    drug_description_map = {}
    fail_drug_des = []
    for csv_path in csv_paths :
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        #print(df.columns.tolist())
        # Extract unique UniProt IDs from the 'Gene' column
        unique_drugs = df['DrugBank ID'].unique()

        # Retrieve gene functions for each unique UniProt ID
        count = 0
        for drug in unique_drugs :
            if drug not in drug_description_map.keys() :
                fail_des = 0
                try :
                #print(drug)
                #print(drug_df.loc[drug_df['drugbank_ids']==drug]['description'].unique())
                #print(drug_df.loc[drug_df['drugbank_ids']==drug]['indication'].unique())
                    description = drug_df.loc[drug_df['drugbank_ids']==drug]['description'].unique()[0] + drug_df.loc[drug_df['drugbank_ids']==drug]['indication'].unique()[0]
                    drug_description_map[drug] = description
                    drug_smiles_map[drug] = df.loc[df['DrugBank ID']==drug]['SMILES'].unique()[0]
                    count+=1
                    if count % 100 == 0 and count > 0 :
                        print('Retrieved 100 gene description')
                    if count % 10000 == 0 or count == 1 :
                        print("Line 135, ", drug, description)
                except :
                    fail_drug_des.append(drug)
                    drug_description_map[drug] = "Default description"
                    print("Not found", drug, "in the drug feature csv file.")
            else :
                print(drug, "in", csv_path, "already exists")
        print("Fail des : ", fail_des)
        print("Retrieval done")
    res = [drug_smiles_map, drug_description_map]
    print("Fail drug des : ", fail_drug_des, len(fail_drug_des))
    for i, output_json_path in enumerate(output_json_paths) :
        with open(output_json_path, 'w') as f :
            json.dump(res[i], f, indent=4)
            print("Write to ", output_json_path)
        
            
    return drug_description_map, drug_smiles_map

def get_gene2seqfunc(csv_paths, output_json_paths):
    # Get sequence and function from GeneID.
    gene_sequence_map = {}
    gene_function_map = {}
    for csv_path in csv_paths :
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        #print(df.columns.tolist())
        # Extract unique UniProt IDs from the 'Gene' column
        unique_genes = df['Gene'].unique()

        # Retrieve gene functions for each unique UniProt ID
        count = 0
        for gene in unique_genes :
            if gene not in gene_sequence_map.keys() :
               function = df.loc[df['Gene'] == gene]['gene_function'].unique()[0]
               sequence = df.loc[df['Gene'] == gene]['Target Sequence'].unique()[0]
               gene_sequence_map[gene] = sequence
               gene_function_map[gene] = function
               count+=1
               if count % 100 == 0 and count > 0 :
                print('Retrieved 100 gene description')
               if count % 10000 == 0 or count == 1 :
                print("Line 135, ", gene, sequence, function)               
            else :
                print(gene, "in", csv_path, "already exists")
        print("Retrieval done")
    res = [gene_function_map, gene_sequence_map]
    for i, output_json_path in enumerate(output_json_paths) :
        with open(output_json_path, 'w') as f :
            json.dump(res[i], f, indent=4)
            print("Write to ", output_json_path)
    return gene_function_map, gene_sequence_map
#gene_function_map, gene_sequence_map = get_gene2seqfunc(output_csv_paths_new, output_json_paths_gene)

def protbert_embeddings(gene2sequence, pretrained_seq_embed_path) :
    tokenizer, model = get_protbert()  # Get protein sequence representation model
    model = model.to(device)
    
    # Tokenize all sequences, ensuring outputs are tensors
    sequences = list(gene2sequence.values())
    inputs = tokenizer(sequences, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to the device
    
    batch_size = 30
    num_inputs = inputs['input_ids'].shape[0]
    num_batch = (num_inputs + batch_size - 1) // batch_size  # Calculate number of batches
    all_embeddings = []
    
    # Process batches
    with torch.no_grad():
        for i in range(num_batch):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_inputs)
            batch_inputs = {k: v[batch_start:batch_end] for k, v in inputs.items()}
            
            # Replace invalid amino acids in input_ids if necessary
            outputs = model(**batch_inputs)
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
            batch_embeddings = last_hidden_states.mean(dim=1)  # Take mean over sequence length
            print(f'{batch_end} elements processed out of {num_inputs}')
            all_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    print(embeddings.shape)
    
    # Create a dictionary for UniProtID -> BioBERT embedding
    test_keys = list(gene2sequence.keys())
    res = {uniprot_id: embeddings[i].tolist() for i, uniprot_id in enumerate(test_keys)}
    
    # Save the embeddings dictionary
    try:
        with open(pretrained_seq_embed_path, 'w') as f:
            json.dump(res, f, indent=4)
        print(f"BioBERT embeddings saved to {pretrained_seq_embed_path}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
    
    return gene2sequence, embeddings

def molformer_embeddings(drug2smiles, pretrained_seq_embed_path) :
    tokenizer, model = get_molformer() # get protein sequence representation model
    model = model.to(device)
    inputs = tokenizer(list(drug2smiles.values()), padding=True, return_tensors="pt")
    print(inputs) # A dict, keys () : input_ids, ...;
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_size = 30
    num_inputs = inputs['input_ids'].shape[0]
    num_batch = num_inputs//batch_size + int(num_inputs % batch_size != 0)
    all_embeddings = []
    with torch.no_grad():
        for i in range(num_batch) :
            batch_inputs = {k : v[(i * batch_size):min((i+1)*batch_size, num_inputs)] for k, v in inputs.items()}
            #batch_inputs = re.sub(r"[UZOB]", "X", batch_inputs)
            outputs = model(**batch_inputs)
            batch_embeddings = outputs.pooler_output  # (batch_size, hidden_dim)
            print(f'{min((i+1)*batch_size, num_inputs)} elements passed through the model')
            all_embeddings.append(batch_embeddings)
    embeddings = torch.cat(all_embeddings, dim=0)
    print(embeddings)

    test_keys = list(drug2smiles.keys())
    res = {uniprot_id: embeddings[i].tolist() 
        for i, uniprot_id in enumerate(test_keys)}

    # Save the resulting dictionary to the specified file path
    try :
      with open(pretrained_seq_embed_path, 'w') as f:
          #print(106)
          json.dump(res, f, indent=4)
          #print(108)
      print(f"BioBERT embeddings saved to {pretrained_seq_embed_path}")
      return drug2smiles, embeddings
    except :
      print("Can not save embed to the path !")
      return drug2smiles, embeddings


if __name__ =='__main__' :
    code_folder = os.path.dirname(__file__)
    kgcnh_folder = os.path.dirname(code_folder)
    kgdrp_folder = os.path.dirname(kgcnh_folder)
    """
    DAVIS pretrained sequence embedding
    """
    davis_id2seq_path = os.path.join(kgcnh_folder, 'data', 'DAVIS', 'davis_proteins.json')
    davis_pretrained_seq_embed_path = os.path.join(kgdrp_folder, 'embeddings', 'DAVIS', 'gene', 'gene_sequence.json')
    davis_id2smiles_path = os.path.join(kgcnh_folder, 'data', 'DAVIS', 'davis_ligands_can.json')
    with open(davis_id2seq_path, 'r') as f :
       prot2seq = json.load(f)

    #gene2sequence, sequence_embeddings = protbert_embeddings(prot2seq, davis_pretrained_seq_embed_path)
    
    with open(davis_id2smiles_path, 'r') as f :
        davis_drug2smiles = json.load(f)
    davis_pretrained_smiles_embed_path = os.path.join(kgdrp_folder, 'embeddings', 'DAVIS' ,'pretrained', 'drug', 'drug_smiles.json')
    #drug2smiles, embeddings = molformer_embeddings(davis_drug2smiles, davis_pretrained_smiles_embed_path)

    """
    KIBA
    """
    kiba_id2seq_path = os.path.join(kgcnh_folder, 'data', 'KIBA', 'protein_to_sequence.json')
    kiba_pretrained_seq_embed_path = os.path.join(kgdrp_folder, 'embeddings', 'KIBA', 'pretrained', 'gene', 'gene_sequence.json')
    kiba_id2smiles_path = os.path.join(kgcnh_folder, 'data', 'KIBA', 'ligand_to_smiles.json')
    with open(kiba_id2seq_path, 'r') as f :
       prot2seq = json.load(f)

    gene2sequence, sequence_embeddings = protbert_embeddings(prot2seq, kiba_pretrained_seq_embed_path)
    
    with open(kiba_id2smiles_path, 'r') as f :
        kiba_drug2smiles = json.load(f)
    kiba_pretrained_smiles_embed_path = os.path.join(kgdrp_folder, 'embeddings', 'KIBA' ,'pretrained', 'drug', 'drug_smiles.json')
    drug2smiles, embeddings = molformer_embeddings(kiba_drug2smiles, kiba_pretrained_smiles_embed_path)
    
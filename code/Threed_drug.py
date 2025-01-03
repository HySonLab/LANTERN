import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
import torch
from scipy.spatial import distance_matrix
import pandas as pd
import json
from collections import defaultdict
from ogb.utils.mol import smiles2graph
from torch import nn
from torch_geometric.graphgym.models import AtomEncoder, BondEncoder
from torch_geometric.utils import to_dense_batch


code_folder = os.path.dirname(__file__)
kgcnh_folder = os.path.dirname(code_folder)
kgdrp_folder = os.path.dirname(kgcnh_folder)
raw_dict = os.path.join(kgcnh_folder, 'raw_dict')
drugs2smiles_dict_path = os.path.join(raw_dict, 'drug', 'kg_drugs.json')
drug2_struc_embed_path = os.path.join(kgdrp_folder, 'embeddings', 'pretrained', 'drug', 'drug2structure.json')
no_conformer_drug_path = 'no_conformer_drug.txt'


def smiles_to_graph(smiles) :
    graph = smiles2graph(smiles)
    return Data(x = torch.tensor(graph['node_feat']), edge_index = torch.tensor(graph['edge_index']), edge_attr = torch.tensor(graph['edge_feat']))

#ligand_graph = ligand_graph.to(device)

class LigandGraphEncoder(nn.Module):
    def __init__(self, emb_dim = 32, hidden_dim = 128, num_layer = 3):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layer = num_layer
        for i in range(self.num_layer):
            if i == 0:
                node_fdim = emb_dim
            else:
                node_fdim = hidden_dim
            self.convs.append(GATConv(
                in_channels = node_fdim, out_channels = hidden_dim, 
                heads = 4, concat = False, dropout = 0.1, edge_dim = emb_dim
            ))
            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

    def forward(self, lig_graph):
        lig_graph = self.atom_encoder(lig_graph)
        lig_graph = self.bond_encoder(lig_graph)
        x = lig_graph.x
        for i in range(self.num_layer):
            x = self.convs[i](x, lig_graph.edge_index, lig_graph.edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
        return global_mean_pool(x, lig_graph.batch)
def radius_graph(positions, r):
    pos_array = positions.numpy()
    dist_matrix = distance_matrix(pos_array, pos_array)
    edge_index = torch.nonzero(torch.tensor(dist_matrix) < r, as_tuple=False).T
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove self-loops
    return edge_index


import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Pipeline from SMILES to 128-dimensional embedding
def smiles_to_embedding(smiles, model, device='cuda:0'):
    data = smiles_to_graph(smiles)
    if data is None :
        return None
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(device)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        embedding = model(data)
    return embedding

# Usage Example
    
#Obtain the drug structure representation from the drugbank_id
    
def drug_structure_representation(drugbank_id) :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LigandGraphEncoder().to(device)
    model.eval()
    with open(drugs2smiles_dict_path, 'r') as f :
        drug_dict = json.load(f)
    smiles = drug_dict.get(drugbank_id)
    #drugs2smiles_dict = pd.read_csv(drugs2smiles_dict_path)
    #smiles = drugs2smiles_dict_path[drugbank_id]
    embedding = smiles_to_embedding(smiles, model, device)
    #print("128-Dimensional Embedding:", embedding.shape, '\n', embedding)
    return embedding

def save_drug_structure_representation(drugbank_ids_path) :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LigandGraphEncoder().to(device)
    #os.makedirs(os.path.dirname(drug2_struc_embed_path), exist_ok=True)
    with open(drugbank_ids_path, 'r') as f :
        drug_dict = json.load(f)
    drug2structure = defaultdict(list)
    #with open(drug2_struc_embed_path, 'a') as f :
    no_conformer_drug = []
    for drug in drug_dict :
        smile = drug_dict[drug]
        embedding = smiles_to_embedding(smile, model, device)
        if embedding is not None :
            drug2structure[drug] = embedding.view(-1).tolist()
        else :
            no_conformer_drug.append(drug)
        #break
    with open(drug2_struc_embed_path, 'w') as f :
        json.dump(drug2structure, f, indent=4)
        print("Saved structure representation of drugs to ", drug2_struc_embed_path)
    with open(no_conformer_drug_path, 'w') as f :
        f.write('\n'.join(no_conformer_drug))
        print(f"Saved {len(no_conformer_drug)} drugs with no conformer to ", no_conformer_drug_path)

#save_drug_structure_representation(drugs2smiles_dict_path)




"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LigandGraphEncoder().to(device)
smiles_string = "CC(S)C1=C2C=C3C(C)=C(C(C)S)C4=[N+]3[Fe--]35N6C(=C4)C(C)=C(CCC(O)=O)C6=CC4=[N+]3C(=CC(N25)=C1C)C(C)=C4CCC(O)=O"
embedding = smiles_to_embedding(smiles_string, model, device)
print("3D Embedding:\n", embedding, '\n', embedding.shape)
"""
"""
#end_time = time.time()
#print(end_time-start_time)
#embedding = drug_structure_representation('DB00468')
#print("128-Dimensional Embedding:", '\n', embedding)
# Example SMILES string (Aspirin)
"""




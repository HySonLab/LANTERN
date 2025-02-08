import copy
import random
import torch
import numpy as np
from collections import defaultdict
import os
from torch.utils.data import Dataset
import tqdm
import pandas as pd
import pickle

code_folder = os.path.dirname(__file__)
lantern_folder = os.path.dirname(code_folder)

# Class for processing the relations
class DataProcessor(object):
    def __init__(self, data_path=os.path.join(lantern_folder, 'data', 'BioSNAP'), mode="train") -> None:
        self.root = data_path
        self.mode = mode + '.csv'
        self.sample_subset = None
        self.entities2id = self._get_entities_id()
        self._drug_protein_num = None
        self._drug_num = None
        self._protein_num = None
        #self.indices2entities = self.get_indices2entities()
        

    @property
    def drug_protein_num(self):
        if self._drug_protein_num is None:
            self._calc_drug_protein_num()
        return self._drug_protein_num

    @property
    def _get_protein_num(self):
        if self._protein_num is None:
            self._calc_drug_protein_num()
        return self._protein_num

    @property
    def drug_num(self):
        if self._drug_num is None:
            self._calc_drug_protein_num()
        return self._drug_num

    def get_relation_num(self):
        return len(self.get_relations2id(self))

    def get_node_num(self):
        return len(self.entities2id)

    def get_entities2id(self):
        return self.entities2id

    def get_relations2id(self):
        return self.relations2id
    
    def triple2id(self, triple):
        h, r, t = triple
        h = self.entities2id[h]
        r = int(r)
        t = self.entities2id[t]
        return h, r, t

    def _calc_drug_protein_num(self) :
        res = self._load_data()
        drug_num = -1
        max_id = -1
        for triple in res:
            h, r, t = self.triple2id(triple)
            drug_num = max(drug_num, h)
            max_id = max(max_id, h, t)
        self._drug_protein_num = max_id + 1
        self._drug_num = drug_num + 1
        self._protein_num = max_id - drug_num

    def load_data(self): # Get all triples converted to indices from txt file
        path = os.path.join(self.root, self.mode)
        relation_frame = pd.read_csv(path)
        drug_id_col_name = None
        label_name = None
        gene_id_col_name = None
        if 'BioSNAP' in self.root :
            drug_id_col_name = 'DrugBank ID'
            label_name = 'Label'
            gene_id_col_name = 'Gene'
        elif 'DAVIS' in self.root :
            drug_id_col_name = 'ligand_id'
            label_name = 'Label'
            gene_id_col_name = 'protein_id'
        elif 'KIBA' in self.root :
            drug_id_col_name = 'CHEMBLID'
            label_name = 'Label'
            gene_id_col_name = 'ProteinID'
        elif 'DeepDDI' in self.root :
            drug_id_col_name = 'smiles_1_id'
            label_name = 'label'
            gene_id_col_name = 'smiles_2_id'
        elif 'yeast' in self.root :
            drug_id_col_name = 'protein_1_id'
            label_name = 'label'
            gene_id_col_name = 'protein_2_id'
        else :
            raise ValueError("Not supported dataset name")
        res = relation_frame[[drug_id_col_name, label_name, gene_id_col_name]] # Filter all observed interactions and access two coloumn
        #res[drug_id_col_name] = res[drug_id_col_name].astype(str)
        #res[gene_id_col_name] = res[gene_id_col_name].astype(str)
        res = res.values.tolist()
        
        triples = []
        triples = [self.triple2id(tuple(triple)) for triple in res]
        return triples
    
    def _load_data(self) :
        path = os.path.join(self.root, self.mode, )
        relation_frame = pd.read_csv(path)
        drug_id_col_name = None
        label_name = None
        gene_id_col_name = None
        if 'BioSNAP' in self.root :
            drug_id_col_name = 'DrugBank ID'
            label_name = 'Label'
            gene_id_col_name = 'Gene'
        elif 'DAVIS' in self.root :
            drug_id_col_name = 'ligand_id'
            label_name = 'Label'
            gene_id_col_name = 'protein_id'
        elif 'KIBA' in self.root :
            drug_id_col_name = 'CHEMBLID'
            label_name = 'Label'
            gene_id_col_name = 'ProteinID'
        elif 'DeepDDI' in self.root :
            drug_id_col_name = 'smiles_1_id'
            label_name = 'label'
            gene_id_col_name = 'smiles_2_id'
        elif 'yeast' in self.root :
            drug_id_col_name = 'protein_1_id'
            label_name = 'label'
            gene_id_col_name = 'protein_2_id'
        else :
            raise ValueError("Not supported dataset name")

        res = relation_frame[[drug_id_col_name, label_name, gene_id_col_name]] # Filter all observed interactions and access two coloumn
        res = res.values.tolist()

        return res

    def _get_entities_id(self):
        entities2id = self._create_entities_id()
        return entities2id

    def _create_entities_id(self):
        entities_id = 0
        entities2id = {}
        kg_data = self._load_data()

        def add2entity_dict(entity):
            nonlocal entities_id, entities2id
            if entity not in entities2id.keys():
                entities2id[entity] = entities_id
                entities_id = entities_id + 1

        # load drug and protein first
        for data in kg_data:
            h, _, _ = data
            add2entity_dict(h) # drug
        for data in kg_data:
            _, r, t = data
            add2entity_dict(t) # protein
        return entities2id
    
class DDI_DataProcessor(DataProcessor) :
    def _calc_drug_protein_num(self):
        res = self._load_data()
        drug_num = -1
        max_id = -1
        for triple in res:
            h, r, t = self.triple2id(triple)
            drug_num = max(max_id, h, t)
            max_id = max(max_id, h, t)
        self._drug_protein_num = max_id + 1
        self._drug_num = drug_num + 1
        self._protein_num = 0

class PPI_DataProcessor(DataProcessor) :
    def _calc_drug_protein_num(self):
        res = self._load_data()
        protein_num = -1
        max_id = -1
        for triple in res:
            h, r, t = self.triple2id(triple)
            protein_num = max(max_id, h, t)
            max_id = max(max_id, h, t)
        self._drug_protein_num = max_id + 1
        self._protein_num = protein_num + 1
        self._drug_num = 0
        
# Class to deal with triple : anchor, positive, negative. Dealing with sampling.
# We need to sample neg samples for sub_triple;
# We need all_triple to store all positive samples of (h,r), then negative samples sampled are not in pos of (h,r).
class DrugProteinDataSet(Dataset):
    def __init__(self, all_triple, mode='train') -> None:

        super(DrugProteinDataSet, self).__init__()
        self.all_triple = all_triple
        self.mode = mode
        self.triple_dict = defaultdict(list)
        self.num_entities = 0
        self.tail = None
        self.len_tail = 0
        self.head = None
        self.len_head = 0
        self.process_all_triple()
        self.seen_heads = set()
        

    def process_all_triple(self):
        num_entities = 0
        triple_dict = self.triple_dict
        tail = set()
        head = set()
        for triple in self.all_triple : 
            h, r, t = triple
            num_entities = max(num_entities, h, t)
            head.add(h)
            tail.add(t)
            triple_dict[(h, r)].append(t)
        self.num_entities = num_entities + 1
        self.head = np.array(list(head))
        self.tail = np.array(list(tail))
        self.len_head = len(self.head)
        self.len_tail = len(self.tail)
        self.triple_dict = triple_dict
    
    def generate_neg_sample(self, h, r):
        negative_relation = 1 if r == 0 else 0
        negative_samples = self.triple_dict[(h, negative_relation)]
        num_negative_samples = len(negative_samples)
        return negative_samples
    
    def generate_pos_sample(self, h, r) :
        return self.triple_dict[(h,r)]
    def reset_seen_heads(self):
        """
        Reset seen heads at the start of each epoch to allow re-sampling in the next epoch.
        """
        self.seen_heads.clear()

    def __getitem__(self, index) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """
        Instead of returning `None`, skip and keep fetching until finding an unseen head.
        """
        while index < len(self.all_triple):
            triple = self.all_triple[index]
            head, r, tail = triple
            
            if head not in self.seen_heads:
                self.seen_heads.add(head)  # Mark head as seen
                r = 1
                negative_label = 1 - r
                negative_samples = self.generate_neg_sample(head, r)
                positive_samples = self.generate_pos_sample(head, r)

                positive_labels = [r for i in range(len(positive_samples))]
                negative_labels = [negative_label for i in range(len(negative_samples))]

                return (torch.tensor(head), torch.tensor(tail), torch.tensor(positive_samples), 
                        torch.tensor(negative_samples), torch.tensor(positive_labels, dtype=torch.float), 
                        torch.tensor(negative_labels, dtype=torch.float))
            
            # Move to the next index if duplicate is found
            index += 1
    def __len__(self):
        return len(self.all_triple)

class ProteinProteinDataSet(DrugProteinDataSet):
    def process_all_triple(self):
        num_entities = 0
        triple_dict = self.triple_dict
        tail = set()
        head = set()
        for triple in self.all_triple : 
            h, r, t = triple
            num_entities = max(num_entities, h, t)
            head.add(h)
            tail.add(t)
            triple_dict[(h, r)].append(t)
            if self.mode == 'train' :
                triple_dict[(t, r)].append(h)
                head.add(t)
                tail.add(h)
        self.num_entities = num_entities + 1
        self.head = np.array(list(head))
        self.tail = np.array(list(tail))
        self.len_head = len(self.head)
        self.len_tail = len(self.tail)
        self.triple_dict = triple_dict

class DrugDrugDataSet(DrugProteinDataSet):
    def process_all_triple(self):
        num_entities = 0
        triple_dict = self.triple_dict
        tail = set()
        head = set()
        for triple in self.all_triple : 
            h, r, t = triple
            num_entities = max(num_entities, h, t)
            head.add(h)
            tail.add(t)
            triple_dict[(h, r)].append(t)
            if self.mode == 'train' :
                triple_dict[(t, r)].append(h)
                head.add(t)
                tail.add(h)
        self.num_entities = num_entities + 1
        self.head = np.array(list(head))
        self.tail = np.array(list(tail))
        self.len_head = len(self.head)
        self.len_tail = len(self.tail)
        self.triple_dict = triple_dict



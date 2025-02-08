import torch
from torch import nn
from torch.nn import functional as fn
import traceback
from collections import defaultdict
import numpy as np
import os
import json
from utils import get_biobert_representation, random_initialize, normalize_embedding, find_most_similar_drug, custom_binary_cross_entropy_with_logits
from transformer_encoder_layer import TransformerEncoderLayer
#from torch.nn import TransformerEncoderLayer

class ABCModel(nn.Module):

    def __init__(self):
        super(ABCModel, self).__init__()

    @staticmethod
    def _l2_norm(x):
        return (x * x).sum()

    def l2_loss_mean(self, *args):
        loss = 0
        for x in args:
            loss += self._l2_norm(x)
        return loss

    def calc_reg_loss(self, pos_head, pos_tail, neg_tail=None, *args):
        if neg_tail is not None :
            reg_loss = ((pos_head * pos_head).sum() + (pos_tail * pos_tail).sum() + (neg_tail * neg_tail).sum())
            other_loss = 0
            if len(args) > 0:
                other_loss = self.l2_loss_mean(*args)
            reg_loss = reg_loss + other_loss
            batch_size = (2 * (pos_head.shape[0] + neg_tail.shape[0] * neg_tail.shape[1]))
            return reg_loss / batch_size
        else :
            reg_loss = ((pos_head * pos_head).sum() + (pos_tail * pos_tail).sum())
            other_loss = 0
            if len(args) > 0:
                other_loss = self.l2_loss_mean(*args)
            reg_loss = reg_loss + other_loss
            batch_size = (2 * (pos_head.shape[0]))
            return reg_loss / batch_size

    @staticmethod
    def calc_bpr_loss(pos_score, neg_score):
        if pos_score != None and neg_score != None :
            # utilize the broadcast mechanism
            score = pos_score - neg_score    
        elif pos_score == None and neg_score != None :
            score = neg_score
        elif pos_score != None and neg_score == None :
            score = pos_score
        else :
            score = None
            return None
        loss = -fn.logsigmoid(score).mean()
        return loss
    @staticmethod
    def calc_bce_loss(pos_score = None, neg_score=None, mask=None):
        #print(75, 'model initiall pos_score and neg_score : ' , pos_score.shape, neg_score.shape)
        if neg_score is not None:
            # Flatten and apply mask
            pos_score_flat = pos_score.contiguous().view(-1) if pos_score is not None else None
            neg_score_flat = neg_score.contiguous().view(-1) if neg_score.numel() != 0 else None
            if mask is not None:
                if pos_score is not None :
                    pos_mask_flat = mask['pos_t'].contiguous().view(-1)
                    pos_score_valid = pos_score_flat[pos_mask_flat]

                neg_mask_flat = mask['neg_t'].contiguous().view(-1)
                #print(84, 'model', pos_mask_flat.shape, neg_mask_flat.shape, pos_score_flat.shape)
                neg_score_valid = neg_score_flat[neg_mask_flat]
            else:
                pos_score_valid = pos_score_flat
                neg_score_valid = neg_score_flat

            score = torch.cat((pos_score_valid, neg_score_valid)) if pos_score is not None else neg_score_valid
            pos_label = torch.ones_like(pos_score_valid) if pos_score is not None else None
            neg_label = torch.zeros_like(neg_score_valid)
            label = torch.cat((pos_label, neg_label)).to(score.device) if pos_score is not None else neg_label.to(score.device)

            loss = fn.binary_cross_entropy_with_logits(score, label, pos_weight = None)
            return loss
        else:
            # Handle the case where no negative score is provided
            pos_score_flat = pos_score.contiguous().view(-1)
            if mask is not None:
                pos_mask_flat = mask['pos_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_mask_flat]
            else:
                pos_score_valid = pos_score_flat
            pos_label = torch.ones_like(pos_score_valid) if pos_score is not None else None
            loss = fn.binary_cross_entropy_with_logits(pos_score_valid, pos_label)
            return loss

class Model(ABCModel):
    def __init__(self, entity_num, drug_num, protein_num, embed_dim, weight_decay, p_drop, score_fun='dot', device='cpu', modality = 1, dataset_name = 'BioSNAP',
                train_entity2index=None, path_2_pretrained_embedding = None, drug_pretrained_dim=None, gene_sequence_dim = None):

        super(Model, self).__init__()
        self.entity_num = entity_num
        self.drug_num = drug_num # drug : 4400 protein : 2000 ; train entity2index : id -> index
        self.embed_dim = embed_dim
        self.modality = modality
        self.dataset_name = dataset_name
        self.drug_protein_num = drug_num + protein_num
        self.train_entity2index = train_entity2index
        self.drug_pretrained_dim = drug_pretrained_dim
        self.gene_function_dim = drug_pretrained_dim
        self.gene_sequence_dim = gene_sequence_dim
        init = torch.zeros((entity_num, embed_dim))
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(init, gain=gain)
        self.entity_embed = nn.Parameter(init) # 6400 x embed_dim
        self.protein_num = protein_num
        self.weight_decay = weight_decay
        self.p_drop = p_drop
        self.test_indices_to_train_indices = None
        self.score_fun = score_fun
        self.path_2_pretrained_embedding = path_2_pretrained_embedding

        
        self.drug_des_project = nn.Sequential(
            nn.Linear(self.drug_pretrained_dim, self.embed_dim),
            torch.nn.Dropout(p=self.p_drop, inplace=False),
            nn.ReLU(),
            nn.Linear(self.embed_dim, int(self.embed_dim/self.modality)),
        )
        self.drug_smiles_project = nn.Sequential(
            nn.Linear(self.drug_pretrained_dim, self.embed_dim),
            torch.nn.Dropout(p=self.p_drop, inplace=False),
            nn.ReLU(),
            nn.Linear(self.embed_dim, int(self.embed_dim/self.modality)),
        )
        self.gene_function_project = nn.Sequential(
            nn.Linear(self.gene_function_dim, self.embed_dim),
            torch.nn.Dropout(p=self.p_drop, inplace=False),
            nn.ReLU(),
            nn.Linear(self.embed_dim, int(self.embed_dim/self.modality)),
        )
        self.gene_sequence_project = nn.Sequential(
            nn.Linear(self.gene_sequence_dim, self.embed_dim),
            torch.nn.Dropout(p=self.p_drop, inplace=False),
            nn.ReLU(),
            nn.Linear(self.embed_dim, int(self.embed_dim/self.modality)),
        )

        #if self.enable_pretrained_init :
        self.initialize_pretrained_entity_embed()

        self.assign_embed_graph = False
        self.device = device
        self.encoder_layer = TransformerEncoderLayer(d_model = 2 * self.embed_dim, nhead=8)
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model = 2 * self.embed_dim, nhead=8)
        self.predictor1 = nn.Sequential(self.encoder_layer,
                                     #self.encoder_layer,
                                     nn.Linear(2 * self.embed_dim, 1),
                                     )
        #self.predictor1 = nn.Sequential(nn.Linear(2*self.embed_dim,1))
        self.dropout = nn.Dropout(self.p_drop)

    def get_pretrained_embedding(self) :
        if self.path_2_pretrained_embedding is not None :
            drug_smiles_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained', 'drug', 'drug_smiles.json')
            with open(drug_smiles_pretrained_path, 'r') as f :
                drug_smiles_pretrained = json.load(f)
            gene_sequence_pretrained_path = os.path.join(self.path_2_pretrained_embedding, self.dataset_name,'pretrained','gene', 'gene_sequence.json')
            with open(gene_sequence_pretrained_path, 'r') as f :
                gene_sequence_pretrained = json.load(f)

            if self.modality >= 2 :
                drug_des_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name, 'pretrained','drug', 'drug_description.json')
                with open(drug_des_pretrained_path, 'r') as f :
                    drug_des_pretrained = json.load(f)
                gene_des_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained','gene', 'gene_function.json')
                with open(gene_des_pretrained_path, 'r') as f :
                    gene_des_pretrained = json.load(f)

            if self.modality == 3 :
                drug_struct_embed_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained', 'drug', 'drug2structure.json')
                with open(drug_struct_embed_path, 'r') as f :
                    drug_structure_embed = json.load(f)
                prot_structure_embed_path = os.path.join(self.path_2_pretrained_embedding, self.dataset_name,'pretrained', 'gene', 'prot_structure.json')
                with open(prot_structure_embed_path, 'r') as f :
                    prot_structure_embed = json.load(f)
            
            if self.modality == 3 :
                drug_pretrained = (drug_des_pretrained, drug_smiles_pretrained, drug_structure_embed)
                gene_pretrained = (gene_des_pretrained, gene_sequence_pretrained, prot_structure_embed)
            elif self.modality == 2 :
                drug_pretrained = (drug_des_pretrained, drug_smiles_pretrained)
                gene_pretrained = (gene_des_pretrained, gene_sequence_pretrained)
            elif self.modality == 1 :
                drug_pretrained = drug_smiles_pretrained
                #print("MODEL 243, ", list(drug_pretrained.items())[0])
                #print("MODEL 244, " , drug_pretrained['3081361'])
                gene_pretrained = gene_sequence_pretrained
            else :
                raise ValueError("The model only supports 1, 2 or 3 modalities !")

            return drug_pretrained, gene_pretrained

        else :
            raise ValueError("No pretrained embedding for entities and relations")
        
    def initialize_pretrained_entity_embed(self) :
        if self.path_2_pretrained_embedding is not None :
            train_set_indices2entities = {value : key for key, value in self.train_entity2index.items()}
            drug_pretrained, gene_pretrained = self.get_pretrained_embedding()
            if self.modality == 3 :
                drug_des_pretrained, drug_smiles_pretrained, drug_structure_embed = drug_pretrained
                gene_func_pretrained, gene_sequence_pretrained, prot_structure_embed = gene_pretrained
            elif self.modality == 2 :
                drug_des_pretrained, drug_smiles_pretrained = drug_pretrained
                gene_func_pretrained, gene_sequence_pretrained = gene_pretrained
            elif self.modality == 1 :
                drug_smiles_pretrained = drug_pretrained
                gene_sequence_pretrained = gene_pretrained
            entities = list(train_set_indices2entities.values())
            drugs = entities[:self.drug_num]
            self.trained_drugs = drugs
            genes = entities[self.drug_num:]
            self.trained_genes = genes
     
            # Assuming drugs and genes lists are already defined.
            drug_des_embeddings = []
            drug_smiles_embeddings = []
            drug_structure_embedddings = []
            gene_func_embeddings = []
            gene_sequence_embeddings = []
            pro_structure_embeddings = []

            for drug in drugs:
                drug = str(drug)
                try:
                    #print(283, 'MODEL', type(drug))
                    smiles_embedding = torch.tensor(drug_smiles_pretrained[drug], dtype=torch.float)
                    drug_smiles_embeddings.append(smiles_embedding)
                    if self.modality >= 2 :
                        des_embedding = torch.tensor(drug_des_pretrained[drug], dtype=torch.float)
                        drug_des_embeddings.append(des_embedding)
                    if self.modality == 3 :
                        if drug in drug_structure_embed :
                            drug_structure_embeddding = torch.tensor(drug_structure_embed[drug], dtype=torch.float)
                        else :
                            drug_structure_embeddding = self.drug_smiles_project(smiles_embedding)
                        drug_structure_embedddings.append(drug_structure_embeddding)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    
            # Iterate through genes to get their embeddings
            for gene in genes:
                gene = str(gene)
                try :
                    sequence_embedding = torch.tensor(gene_sequence_pretrained[gene], dtype=torch.float)
                    gene_sequence_embeddings.append(sequence_embedding)
                    if self.modality >= 2 :
                        func_embedding = torch.tensor(gene_func_pretrained[gene], dtype=torch.float)
                        gene_func_embeddings.append(func_embedding)
                    if self.modality == 3 :
                        if gene in prot_structure_embed :
                            pro_structure_embedding = torch.tensor(prot_structure_embed[gene], dtype=torch.float)
                        else :
                            pro_structure_embedding = self.gene_sequence_project(sequence_embedding)
                        pro_structure_embeddings.append(pro_structure_embedding)
                except Exception as e :
                    print(e)
                    print(traceback.format_exc())

            drug_smiles_pretrained = torch.stack(drug_smiles_embeddings)
            drug_smiles_pretrained = self.drug_smiles_project(drug_smiles_pretrained)
            gene_sequence_pretrained = torch.stack(gene_sequence_embeddings)
            gene_sequence_pretrained = self.gene_sequence_project(gene_sequence_pretrained)

            if self.modality >= 2 :
                drug_des_pretrained = torch.stack(drug_des_embeddings)
                drug_des_pretrained = self.drug_des_project(drug_des_pretrained)
                gene_func_pretrained = torch.stack(gene_func_embeddings)
                gene_func_pretrained = self.gene_function_project(gene_func_pretrained)

            if self.modality == 3 :
                drug_structure_pretrained = torch.stack(drug_structure_embedddings)
                pro_structure_pretrained = torch.stack(pro_structure_embeddings)

            if self.modality == 3 :
                drug_pretrained = torch.cat([drug_des_pretrained,drug_smiles_pretrained, drug_structure_pretrained], axis=1)
                gene_pretrained = torch.cat([gene_func_pretrained, gene_sequence_pretrained, pro_structure_pretrained], axis=1)
            elif self.modality == 2 :
                drug_pretrained = torch.cat([drug_des_pretrained,drug_smiles_pretrained], axis=1)
                gene_pretrained = torch.cat([gene_func_pretrained, gene_sequence_pretrained], axis=1)
            elif self.modality == 1 :
                drug_pretrained = drug_smiles_pretrained
                gene_pretrained = gene_sequence_pretrained
            else :
                raise ValueError("The model only supports 1, 2 or 3 modalities !")

            embed = torch.cat([drug_pretrained, gene_pretrained], axis=0)
            self.entity_embed = nn.Parameter(embed, requires_grad = True)

    def project_pretrained_gene_seq(self, x) :
        return self.gene_sequence_project(x)
    
    def project_pretrained_gene_func(self, x) :
        return self.gene_function_project(x)

    def calc_dot_score(self, h_embed, pos_t_embed, n_t_embed=None) :
        neg_score = None
        def calc_score(h_embed, t_embed) :
            if self.modality >= 2 :
                h_nlp_embed = h_embed[:,:,:int(self.embed_dim/self.modality)]
                t_nlp_embed = t_embed[:,:,:int(self.embed_dim/self.modality)]
            h_bio_embed = h_embed[:,:,int(self.embed_dim/self.modality):int(2*self.embed_dim/self.modality)]
            t_bio_embed = t_embed[:,:,int(self.embed_dim/self.modality):int(2*self.embed_dim/self.modality)]
            if self.modality == 3 :
                h_geo_embed = h_embed[:,:,int(2*self.embed_dim/self.modality):]
                t_geo_embed = t_embed[:,:,int(2*self.embed_dim/self.modality):]
            if self.modality == 3 :
                return ((h_nlp_embed*t_nlp_embed).sum(-1))+((h_bio_embed*t_bio_embed).sum(-1))+((h_geo_embed*t_geo_embed).sum(-1))
            elif self.modality == 2 :
                return ((h_nlp_embed*t_nlp_embed).sum(-1))+((h_bio_embed*t_bio_embed).sum(-1))
            else :
                return ((h_bio_embed*t_bio_embed).sum(-1))
        pos_score = calc_score(h_embed, pos_t_embed)
        if n_t_embed is not None :
            neg_score = calc_score(h_embed, n_t_embed) 
        return pos_score, neg_score
    
    def calc_trans_score(self, h_embed, pos_t_embed=None, n_t_embed=None):
        pos_score = None
        neg_score = None
        def calc_score(h_embed, pos_t_embed) :
            if h_embed.shape[1] < pos_t_embed.shape[1] :
                num_pos = pos_t_embed.shape[1]
                h_embed = h_embed.repeat(1, num_pos, 1)
            else :
                num_heads = h_embed.shape[1]
                pos_t_embed = pos_t_embed.repeat(1, num_heads, 1)
            h_pos_embed = torch.cat([h_embed, pos_t_embed],dim=-1)
            num_pos = h_pos_embed.shape[1]
            scores = []
            for i in range(num_pos) :
                v_f = h_pos_embed[:,i,:]
                score = None
                for i, l in enumerate(self.predictor1):
                    if i==(len(self.predictor1)-1):
                        feat = v_f
                        score = l(v_f)
                    else:
                        v_f = l(v_f)
                        v_f = fn.relu(self.dropout(v_f))
                scores.append(score)
            scores = torch.cat(scores, dim = 1)
            return scores
        if pos_t_embed.numel() != 0 :
            pos_score = calc_score(h_embed, pos_t_embed)
        if n_t_embed.numel() != 0 :
            neg_score = calc_score(h_embed, n_t_embed)
        return pos_score, neg_score

    def train_step(self, h, t, pos_t, n_t, mask=None):
        pos_score, neg_score, embed = self.__forward_wo_augment(h, t, pos_t, n_t)
        if pos_score is None and neg_score is None :
            loss = None
        else :
            loss = self.calc_bce_loss(pos_score, neg_score, mask=mask) # , mask=mask

        if n_t.numel() == 0 :
            if pos_t.numel() == 0 :
                reg_loss = None
            else :
                reg_loss = self.calc_reg_loss(embed[h], embed[pos_t]) # , mask = mask
        else :
            #print(245, "model", n_t)
            reg_loss = self.calc_reg_loss(embed[h], embed[pos_t], embed[n_t]) #

        reg_loss = self.weight_decay * reg_loss
        #print(368, "model", "loss : ", loss, "reg_loss : ", reg_loss)
        return loss + reg_loss, reg_loss
    
    def __forward_wo_augment(self, h, t, pos_t, n_t):
        embed = self.entity_embed
        h_embed = self.entity_embed[h]
        h_embed = h_embed.unsqueeze(1) # bs, 1, embed_dim
        pos_t_embed = self.entity_embed[pos_t]
        n_t_embed = self.entity_embed[n_t]
        if self.score_fun == 'transformer':
            pos_score, neg_score = self.calc_trans_score(h_embed,pos_t_embed,n_t_embed)
        else:
            pos_score, neg_score = self.calc_dot_score(h_embed,pos_t_embed,n_t_embed)
        return pos_score, neg_score, embed

    def predict(self, h, pos_t, n_t, test_entity2index, mask = None) :
        device = h.device
        test_index2entity = {value : key for key, value in test_entity2index.items()}
        drug_test = [test_index2entity[id.item()] for id in h]
        #print(448, 'MODEL', drug_test)
        try :
            #print(450, 'MODEL', self.train_entity2index)
            indices = [self.train_entity2index[(test_entity)] for test_entity in drug_test]
            h_embed = self.entity_embed[indices]
        except :
            h_embed = torch.stack([self.get_trained_entity_embed(drug, type='drug').to(self.device) for drug in drug_test])
        h_embed = h_embed.unsqueeze(1)

        batch_pos_t_embed = []
        batch_neg_t_embed = []

        for pos_id in pos_t: # each batch
            try :
                pos_gene = [test_index2entity[pos_gene.item()] for pos_gene in pos_id]
                indices = [self.train_entity2index[entity] for entity in pos_gene]
                pos_t_embed = self.entity_embed[indices]
            except :
                pos_t_embed = []
                for pos_gene in pos_id :
                    pos_gene = test_index2entity[pos_gene.item()]
                    pos_gene_embed = self.get_trained_entity_embed(pos_gene, type='gene')
                    pos_t_embed.append(pos_gene_embed.to(device))
                pos_t_embed = torch.stack(pos_t_embed).to(device)
            batch_pos_t_embed.append(pos_t_embed)
    
        batch_pos_t_embed = torch.stack(batch_pos_t_embed)
        
        if n_t.numel() != 0 :

            for neg_id in n_t :
                try :
                    neg_gene = [test_index2entity[neg_gene.item()] for neg_gene in neg_id]
                    indices = [self.train_entity2index[entity] for entity in neg_gene]
                    neg_t_embed = self.entity_embed[indices]
                except :
                    neg_t_embed = []
                    for neg_gene in neg_id :
                        neg_gene = neg_gene.item()
                        neg_gene = test_index2entity[neg_gene]
                        neg_gene_embed = self.get_trained_entity_embed(neg_gene, type='gene')
                        neg_t_embed.append(neg_gene_embed.to(device))
                    neg_t_embed = torch.stack(neg_t_embed).to(device)
                batch_neg_t_embed.append(neg_t_embed)
            batch_neg_t_embed = torch.stack(batch_neg_t_embed)
        batch_neg_t_embed = batch_neg_t_embed.to(device) if isinstance(batch_neg_t_embed,torch.Tensor) else None
        batch_pos_t_embed = batch_pos_t_embed.to(device) if isinstance(batch_pos_t_embed, torch.Tensor) else None
        
        if n_t.numel() != 0 :
            if self.score_fun == 'transformer' :
                pos_score, neg_score = self.calc_trans_score(h_embed, batch_pos_t_embed, batch_neg_t_embed)
            else :
                pos_score, neg_score = self.calc_dot_score(h_embed, batch_pos_t_embed, batch_neg_t_embed)
            pos_score_flat = pos_score.contiguous().view(-1) if pos_score is not None else None
            neg_score_flat = neg_score.contiguous().view(-1)
            

            if mask is not None :
                pos_flat_mask = mask['pos_t'].contiguous().view(-1) if pos_score is not None else None
                neg_flat_mask = mask['neg_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_flat_mask] if pos_score is not None else None
                neg_score_valid = neg_score_flat[neg_flat_mask]

                score = torch.cat([pos_score_valid, neg_score_valid],dim=-1)if pos_score is not None  else neg_score_valid

            else :
  
                score = torch.cat([pos_score_flat, neg_score_flat], dim=-1)
            return score
        elif n_t.numel() == 0 :
            if pos_t.numel() == 0 :
                return None
            if self.score_fun == 'transformer' :
                pos_score, _ = self.calc_trans_score(h_embed, batch_pos_t_embed, batch_pos_t_embed)
            else :
                pos_score, _ = self.calc_dot_score(h_embed, batch_pos_t_embed, batch_pos_t_embed)
            pos_score_flat = pos_score.contiguous().view(-1)
            if mask is not None :
                pos_flat_mask = mask['pos_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_flat_mask]
                return pos_score_valid
            return pos_score_flat
        
        
    def get_trained_entity_embed(self, test_entity, type='drug') :
        """
        Get trained entity embedding for predictions from the ID of drugs or genes
        """
        #    return random_initialize((self.embed_dim, ))
        if test_entity in self.train_entity2index.keys() :
            return self.entity_embed[self.train_entity2index[test_entity]]
        else :
            drug_pretrained, gene_pretrained = self.get_pretrained_embedding()
            if self.modality == 3 :
                drug_des_pretrained_dict, drug_smiles_pretrained_dict, drug_structure_embed = drug_pretrained
                gene_func_pretrained_dict, gene_sequence_pretrained_dict, prot_structure_embed = gene_pretrained
            elif self.modality == 2 :
                drug_des_pretrained_dict, drug_smiles_pretrained_dict = drug_pretrained
                gene_func_pretrained_dict, gene_sequence_pretrained_dict = gene_pretrained
            elif self.modality == 1 :
                drug_smiles_pretrained_dict = drug_pretrained
                gene_sequence_pretrained_dict = gene_pretrained

            if type=='drug' :
            # Obtain the pretrained embeddings of drugs and genes
                try :
                    drug_smiles_pretrained = torch.tensor(drug_smiles_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                    trained_drug_smiles_pretrained_embed = {
                        drug : torch.tensor(drug_smiles_pretrained_dict[drug]).to(self.device) for drug in self.trained_drugs
                    }
                    most_similar_drug_smiles, sim_smiles = find_most_similar_drug(drug_smiles_pretrained, trained_drug_smiles_pretrained_embed)
                    #print(543, 'model', most_similar_drug_smiles)
                    drug_smiles_pretrained = self.drug_smiles_project(drug_smiles_pretrained)
                    #drug_smiles = (drug_smiles_pretrained + (self.entity_embed[self.train_entity2index[most_similar_drug_smiles]])[int(self.embed_dim/self.modality):int(2 * self.embed_dim/self.modality)])/2
                    drug_smiles = drug_smiles_pretrained

                    if self.modality >= 2 :
                        drug_des_pretrained = torch.tensor(drug_des_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                        drug_des_pretrained = self.drug_des_project(drug_des_pretrained)  # pretrained dim -> embed dim
                        drug_des = drug_des_pretrained
                    
                    if self.modality == 3 :
                        if test_entity in drug_structure_embed :
                            drug_structure_embeddding = torch.tensor(drug_structure_embed[test_entity], dtype=torch.float).to(self.device)
                        else :
                            drug_structure_embeddding = self.drug_smiles_project(drug_smiles_pretrained).to(self.device)
                    if self.modality == 3 :
                        embed = torch.cat([drug_des,drug_smiles,drug_structure_embeddding], axis=-1)
                    elif self.modality == 2 :
                        embed = torch.cat([drug_des,drug_smiles], axis=-1)
                    else :
                        embed = drug_smiles
                    return embed
                    
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
            else :
                try :

                    trained_gene_seq_pretrained_embed = {
                        gene : torch.tensor(gene_sequence_pretrained_dict[gene]).to(self.device) for gene in self.trained_genes
                    }
                    gene_sequence_pretrained = torch.tensor(gene_sequence_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                    most_similar_gene_seq, sim_seq = find_most_similar_drug(gene_sequence_pretrained, trained_gene_seq_pretrained_embed)
                    gene_sequence_pretrained = self.gene_sequence_project(gene_sequence_pretrained) # OLD VERSION
                    gene_seq = gene_sequence_pretrained

                    if self.modality >= 2 :
                        trained_gene_func_pretrained_embed = {
                            gene : torch.tensor(gene_func_pretrained_dict[gene]).to(self.device) for gene in self.trained_genes
                        }
                        gene_func_pretrained = torch.tensor(gene_func_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                        most_similar_gene_func, sim_func = find_most_similar_drug(gene_func_pretrained, trained_gene_func_pretrained_embed)
                        gene_func_pretrained = self.gene_function_project(gene_func_pretrained) # OLD VERSION
                        gene_func = gene_func_pretrained

                    if self.modality == 3 :
                        if test_entity in prot_structure_embed:
                            prot_structure_embeddding = torch.tensor(prot_structure_embed[test_entity], dtype=torch.float).to(self.device)
                        else :
                            prot_structure_embeddding = self.gene_sequence_project(gene_sequence_pretrained).to(self.device)
                    if self.modality == 3 :
                        embed = torch.cat([gene_func, gene_seq, prot_structure_embeddding], axis=-1)
                    elif self.modality == 2 :
                        embed = torch.cat([gene_func, gene_seq], axis=-1)
                    else :
                        embed = gene_seq
                    return embed
                    
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                
    def get_sequence_project(self, sequence) :
        return self.gene_sequence_project(sequence) # (1, embed_dim/2)
    
    def make_link_predictions(self, h_embed, n_t_embed) :
        """
        Given new/similar protein targets, forward drugs "treat" that protein
        h : potential drugs : (drug_num, 2*embed_size)
        n_t : protein targets : (1, 2 * embed_size)
        """
        if self.score_fun == 'transformer' :
            return self.calc_trans_score(h_embed, n_t_embed)
        else :
            return self.calc_dot_score(h_embed, n_t_embed)

class DDI_Model(Model) :
    def get_pretrained_embedding(self) :
        if self.path_2_pretrained_embedding is not None :
            drug_smiles_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained', 'drug', 'drug_smiles.json')
            with open(drug_smiles_pretrained_path, 'r') as f :
                drug_smiles_pretrained = json.load(f)
            if self.modality >= 2 :
                drug_des_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name, 'pretrained','drug', 'drug_description.json')
                with open(drug_des_pretrained_path, 'r') as f :
                    drug_des_pretrained = json.load(f)
            if self.modality == 3 :
                drug_struct_embed_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained', 'drug', 'drug2structure.json')
                with open(drug_struct_embed_path, 'r') as f :
                    drug_structure_embed = json.load(f)
            if self.modality == 3 :
                drug_pretrained = (drug_des_pretrained, drug_smiles_pretrained, drug_structure_embed)
            elif self.modality == 2 :
                drug_pretrained = (drug_des_pretrained, drug_smiles_pretrained)
            elif self.modality == 1 :
                drug_pretrained = drug_smiles_pretrained
            else :
                raise ValueError("The model only supports 1, 2 or 3 modalities !")
            return drug_pretrained
        else :
            raise ValueError("No pretrained embedding for entities and relations")
        
    def initialize_pretrained_entity_embed(self) :
        if self.path_2_pretrained_embedding is not None :
            train_set_indices2entities = {value : key for key, value in self.train_entity2index.items()}
            drug_pretrained = self.get_pretrained_embedding()
            if self.modality == 3 :
                drug_des_pretrained, drug_smiles_pretrained, drug_structure_embed = drug_pretrained
            elif self.modality == 2 :
                drug_des_pretrained, drug_smiles_pretrained = drug_pretrained
            elif self.modality == 1 :
                drug_smiles_pretrained = drug_pretrained
            entities = list(train_set_indices2entities.values())
            drugs = entities[:self.drug_num]
            self.trained_drugs = drugs
     
            # Assuming drugs and genes lists are already defined.
            drug_des_embeddings = []
            drug_smiles_embeddings = []
            drug_structure_embedddings = []

            for drug in drugs:
                drug = str(drug)
                try:
                    #print(283, 'MODEL', type(drug))
                    smiles_embedding = torch.tensor(drug_smiles_pretrained[drug], dtype=torch.float)
                    drug_smiles_embeddings.append(smiles_embedding)
                    if self.modality >= 2 :
                        des_embedding = torch.tensor(drug_des_pretrained[drug], dtype=torch.float)
                        drug_des_embeddings.append(des_embedding)
                    if self.modality == 3 :
                        if drug in drug_structure_embed :
                            drug_structure_embeddding = torch.tensor(drug_structure_embed[drug], dtype=torch.float)
                        else :
                            drug_structure_embeddding = self.drug_smiles_project(smiles_embedding)
                        drug_structure_embedddings.append(drug_structure_embeddding)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())

            drug_smiles_pretrained = torch.stack(drug_smiles_embeddings)
            drug_smiles_pretrained = self.drug_smiles_project(drug_smiles_pretrained)

            if self.modality >= 2 :
                drug_des_pretrained = torch.stack(drug_des_embeddings)
                drug_des_pretrained = self.drug_des_project(drug_des_pretrained)

            if self.modality == 3 :
                drug_structure_pretrained = torch.stack(drug_structure_embedddings)

            if self.modality == 3 :
                drug_pretrained = torch.cat([drug_des_pretrained,drug_smiles_pretrained, drug_structure_pretrained], axis=1)
            elif self.modality == 2 :
                drug_pretrained = torch.cat([drug_des_pretrained,drug_smiles_pretrained], axis=1)
            elif self.modality == 1 :
                drug_pretrained = drug_smiles_pretrained
            else :
                raise ValueError("The model only supports 1, 2 or 3 modalities !")

            embed = torch.cat([drug_pretrained], axis=0)
            self.entity_embed = nn.Parameter(embed, requires_grad = True)


    def predict(self, h, pos_t, n_t, test_entity2index, mask = None) :
        device = h.device
        test_index2entity = {value : key for key, value in test_entity2index.items()}
        drug_test = [test_index2entity[id.item()] for id in h]
        
        try :
            #print(450, 'MODEL', self.train_entity2index)
            indices = [self.train_entity2index[(test_entity)] for test_entity in drug_test]
            h_embed = self.entity_embed[indices]
        except :
            h_embed = torch.stack([self.get_trained_entity_embed(drug, type='drug').to(self.device) for drug in drug_test])
        h_embed = h_embed.unsqueeze(1)

        batch_pos_t_embed = []
        batch_neg_t_embed = []

        for pos_id in pos_t: # each batch
            try :
                pos_drug = [test_index2entity[pos_drug.item()] for pos_drug in pos_id]
                indices = [self.train_entity2index[entity] for entity in pos_drug]
                pos_t_embed = self.entity_embed[indices]
            except :
                pos_t_embed = []
                for pos_drug in pos_id :
                    pos_drug = test_index2entity[pos_drug.item()]
                    pos_drug_embed = self.get_trained_entity_embed(pos_drug, type='drug')
                    pos_t_embed.append(pos_drug_embed.to(device))
                pos_t_embed = torch.stack(pos_t_embed).to(device)
            batch_pos_t_embed.append(pos_t_embed)
    
        batch_pos_t_embed = torch.stack(batch_pos_t_embed)
        
        if n_t.numel() != 0 :

            for neg_id in n_t :
                try :
                    neg_drug = [test_index2entity[neg_drug.item()] for neg_drug in neg_id]
                    indices = [self.train_entity2index[entity] for entity in neg_drug]
                    neg_t_embed = self.entity_embed[indices]
                except :
                    neg_t_embed = []
                    for neg_drug in neg_id :
                        neg_drug = neg_drug.item()
                        neg_drug = test_index2entity[neg_drug]
                        neg_drug_embed = self.get_trained_entity_embed(neg_drug, type='drug')
                        neg_t_embed.append(neg_drug_embed.to(device))
                    neg_t_embed = torch.stack(neg_t_embed).to(device)
                batch_neg_t_embed.append(neg_t_embed)
            batch_neg_t_embed = torch.stack(batch_neg_t_embed)
        batch_neg_t_embed = batch_neg_t_embed.to(device) if isinstance(batch_neg_t_embed,torch.Tensor) else None
        batch_pos_t_embed = batch_pos_t_embed.to(device) if isinstance(batch_pos_t_embed, torch.Tensor) else None
        
        if n_t.numel() != 0 :
            if self.score_fun == 'transformer' :
                pos_score, neg_score = self.calc_trans_score(h_embed, batch_pos_t_embed, batch_neg_t_embed)
            else :
                pos_score, neg_score = self.calc_dot_score(h_embed, batch_pos_t_embed, batch_neg_t_embed)
            pos_score_flat = pos_score.contiguous().view(-1) if pos_score is not None else None
            neg_score_flat = neg_score.contiguous().view(-1)
            

            if mask is not None :
                pos_flat_mask = mask['pos_t'].contiguous().view(-1) if pos_score is not None else None
                neg_flat_mask = mask['neg_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_flat_mask] if pos_score is not None else None
                neg_score_valid = neg_score_flat[neg_flat_mask]

                score = torch.cat([pos_score_valid, neg_score_valid],dim=-1)if pos_score is not None  else neg_score_valid

            else :
  
                score = torch.cat([pos_score_flat, neg_score_flat], dim=-1)
            return score
        elif n_t.numel() == 0 :
            if pos_t.numel() == 0 :
                return None
            if self.score_fun == 'transformer' :
                pos_score, _ = self.calc_trans_score(h_embed, batch_pos_t_embed, batch_pos_t_embed)
            else :
                pos_score, _ = self.calc_dot_score(h_embed, batch_pos_t_embed, batch_pos_t_embed)
            pos_score_flat = pos_score.contiguous().view(-1)
            if mask is not None :
                pos_flat_mask = mask['pos_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_flat_mask]
                return pos_score_valid
            return pos_score_flat
        
        
    def get_trained_entity_embed(self, test_entity, type='drug') :
        """
        Get trained entity embedding for predictions from the ID of drugs or genes
        """
        if test_entity in self.train_entity2index.keys() :
            return self.entity_embed[self.train_entity2index[test_entity]]
        else :
            drug_pretrained = self.get_pretrained_embedding()
            if self.modality == 3 :
                drug_des_pretrained_dict, drug_smiles_pretrained_dict, drug_structure_embed = drug_pretrained
            elif self.modality == 2 :
                drug_des_pretrained_dict, drug_smiles_pretrained_dict = drug_pretrained
            elif self.modality == 1 :
                drug_smiles_pretrained_dict = drug_pretrained
            if type=='drug' :
            # Obtain the pretrained embeddings of drugs and genes
                try :
                    drug_smiles_pretrained = torch.tensor(drug_smiles_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                    trained_drug_smiles_pretrained_embed = {
                        drug : torch.tensor(drug_smiles_pretrained_dict[drug]).to(self.device) for drug in self.trained_drugs
                    }
                    most_similar_drug_smiles, sim_smiles = find_most_similar_drug(drug_smiles_pretrained, trained_drug_smiles_pretrained_embed)
                    #print(543, 'model', most_similar_drug_smiles)
                    drug_smiles_pretrained = self.drug_smiles_project(drug_smiles_pretrained)
                    #drug_smiles = (drug_smiles_pretrained + (self.entity_embed[self.train_entity2index[most_similar_drug_smiles]])[int(self.embed_dim/self.modality):int(2 * self.embed_dim/self.modality)])/2
                    drug_smiles = drug_smiles_pretrained

                    if self.modality >= 2 :
                        drug_des_pretrained = torch.tensor(drug_des_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                        drug_des_pretrained = self.drug_des_project(drug_des_pretrained)  # pretrained dim -> embed dim
                        drug_des = drug_des_pretrained
                    
                    if self.modality == 3 :
                        if test_entity in drug_structure_embed :
                            drug_structure_embeddding = torch.tensor(drug_structure_embed[test_entity], dtype=torch.float).to(self.device)
                        else :
                            drug_structure_embeddding = self.drug_smiles_project(drug_smiles_pretrained).to(self.device)
                    
                    if self.modality == 3 :
                        embed = torch.cat([drug_des,drug_smiles,drug_structure_embeddding], axis=-1)
                    elif self.modality == 2 :
                        embed = torch.cat([drug_des,drug_smiles], axis=-1)
                    else :
                        embed = drug_smiles
                    return embed
                    
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
            else :
                raise ValueError("DDI Model does not support entity of type other than drugs")
            
class PPI_Model(Model) :
    def get_pretrained_embedding(self) :
        if self.path_2_pretrained_embedding is not None :
            gene_sequence_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained', 'gene', 'gene_sequence.json')
            with open(gene_sequence_pretrained_path, 'r') as f :
                gene_sequence_pretrained = json.load(f)
            if self.modality >= 2 :
                gene_func_pretrained_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name, 'pretrained','gene', 'gene_function.json')
                with open(gene_func_pretrained_path, 'r') as f :
                    gene_func_pretrained = json.load(f)
            if self.modality == 3 :
                gene_struct_embed_path = os.path.join(self.path_2_pretrained_embedding,  self.dataset_name,'pretrained', 'gene', 'prot2structure.json')
                with open(gene_struct_embed_path, 'r') as f :
                    gene_structure_embed = json.load(f)
            if self.modality == 3 :
                gene_pretrained = (gene_func_pretrained, gene_sequence_pretrained, gene_structure_embed)
            elif self.modality == 2 :
                gene_pretrained = (gene_func_pretrained, gene_sequence_pretrained)
            elif self.modality == 1 :
                gene_pretrained = gene_sequence_pretrained
            else :
                raise ValueError("The model only supports 1, 2 or 3 modalities !")
            return gene_pretrained
        else :
            raise ValueError("No pretrained embedding for entities and relations")
    def initialize_pretrained_entity_embed(self) :
        if self.path_2_pretrained_embedding is not None :
            train_set_indices2entities = {value : key for key, value in self.train_entity2index.items()}
            gene_pretrained = self.get_pretrained_embedding()
            if self.modality == 3 :
                gene_func_pretrained, gene_sequence_pretrained, gene_structure_embed = gene_pretrained
            elif self.modality == 2 :
                gene_func_pretrained, gene_sequence_pretrained = gene_pretrained
            elif self.modality == 1 :
                gene_sequence_pretrained = gene_pretrained
            entities = list(train_set_indices2entities.values())
            genes = entities[:self.protein_num]
            self.trained_genes = genes
     
            # Assuming genes and genes lists are already defined.
            gene_func_embeddings = []
            gene_sequence_embeddings = []
            gene_structure_embedddings = []

            for gene in genes:
                gene = str(gene)
                try:
                    #print(283, 'MODEL', type(gene))
                    sequence_embedding = torch.tensor(gene_sequence_pretrained[gene], dtype=torch.float)
                    gene_sequence_embeddings.append(sequence_embedding)
                    if self.modality >= 2 :
                        func_embedding = torch.tensor(gene_func_pretrained[gene], dtype=torch.float)
                        gene_func_embeddings.append(func_embedding)
                    if self.modality == 3 :
                        if gene in gene_structure_embed :
                            gene_structure_embeddding = torch.tensor(gene_structure_embed[gene], dtype=torch.float)
                        else :
                            gene_structure_embeddding = self.gene_sequence_project(sequence_embedding)
                        gene_structure_embedddings.append(gene_structure_embeddding)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())

            gene_sequence_pretrained = torch.stack(gene_sequence_embeddings)
            gene_sequence_pretrained = self.gene_sequence_project(gene_sequence_pretrained)

            if self.modality >= 2 :
                gene_func_pretrained = torch.stack(gene_func_embeddings)
                gene_func_pretrained = self.gene_function_project(gene_func_pretrained)

            if self.modality == 3 :
                gene_structure_pretrained = torch.stack(gene_structure_embedddings)

            if self.modality == 3 :
                gene_pretrained = torch.cat([gene_func_pretrained, gene_sequence_pretrained, gene_structure_pretrained], axis=1)
            elif self.modality == 2 :
                gene_pretrained = torch.cat([gene_func_pretrained,gene_sequence_pretrained], axis=1)
            elif self.modality == 1 :
                gene_pretrained = gene_sequence_pretrained
            else :
                raise ValueError("The model only supports 1, 2 or 3 modalities !")

            embed = torch.cat([gene_pretrained], axis=0)
            self.entity_embed = nn.Parameter(embed, requires_grad = True)

    def predict(self, h, pos_t, n_t, test_entity2index, mask = None) :
        device = h.device
        test_index2entity = {value : key for key, value in test_entity2index.items()}
        gene_test = [test_index2entity[id.item()] for id in h]
        
        try :
            #print(450, 'MODEL', self.train_entity2index)
            indices = [self.train_entity2index[(test_entity)] for test_entity in gene_test]
            h_embed = self.entity_embed[indices]
        except :
            h_embed = torch.stack([self.get_trained_entity_embed(gene, type='gene').to(self.device) for gene in gene_test])
        h_embed = h_embed.unsqueeze(1)

        batch_pos_t_embed = []
        batch_neg_t_embed = []

        for pos_id in pos_t: # each batch
            try :
                pos_gene = [test_index2entity[pos_gene.item()] for pos_gene in pos_id]
                indices = [self.train_entity2index[entity] for entity in pos_gene]
                pos_t_embed = self.entity_embed[indices]
            except :
                pos_t_embed = []
                for pos_gene in pos_id :
                    pos_gene = test_index2entity[pos_gene.item()]
                    pos_gene_embed = self.get_trained_entity_embed(pos_gene, type='gene')
                    pos_t_embed.append(pos_gene_embed.to(device))
                pos_t_embed = torch.stack(pos_t_embed).to(device)
            batch_pos_t_embed.append(pos_t_embed)
    
        batch_pos_t_embed = torch.stack(batch_pos_t_embed)
        
        if n_t.numel() != 0 :

            for neg_id in n_t :
                try :
                    neg_gene = [test_index2entity[neg_gene.item()] for neg_gene in neg_id]
                    indices = [self.train_entity2index[entity] for entity in neg_gene]
                    neg_t_embed = self.entity_embed[indices]
                except :
                    neg_t_embed = []
                    for neg_gene in neg_id :
                        neg_gene = neg_gene.item()
                        neg_gene = test_index2entity[neg_gene]
                        neg_gene_embed = self.get_trained_entity_embed(neg_gene, type='gene')
                        neg_t_embed.append(neg_gene_embed.to(device))
                    neg_t_embed = torch.stack(neg_t_embed).to(device)
                batch_neg_t_embed.append(neg_t_embed)
            batch_neg_t_embed = torch.stack(batch_neg_t_embed)
        batch_neg_t_embed = batch_neg_t_embed.to(device) if isinstance(batch_neg_t_embed,torch.Tensor) else None
        batch_pos_t_embed = batch_pos_t_embed.to(device) if isinstance(batch_pos_t_embed, torch.Tensor) else None
        
        if n_t.numel() != 0 :
            if self.score_fun == 'transformer' :
                pos_score, neg_score = self.calc_trans_score(h_embed, batch_pos_t_embed, batch_neg_t_embed)
            else :
                pos_score, neg_score = self.calc_dot_score(h_embed, batch_pos_t_embed, batch_neg_t_embed)
            pos_score_flat = pos_score.contiguous().view(-1) if pos_score is not None else None
            neg_score_flat = neg_score.contiguous().view(-1)
            

            if mask is not None :
                pos_flat_mask = mask['pos_t'].contiguous().view(-1) if pos_score is not None else None
                neg_flat_mask = mask['neg_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_flat_mask] if pos_score is not None else None
                neg_score_valid = neg_score_flat[neg_flat_mask]

                score = torch.cat([pos_score_valid, neg_score_valid],dim=-1)if pos_score is not None  else neg_score_valid

            else :
  
                score = torch.cat([pos_score_flat, neg_score_flat], dim=-1)
            return score
        elif n_t.numel() == 0 :
            if pos_t.numel() == 0 :
                return None
            if self.score_fun == 'transformer' :
                pos_score, _ = self.calc_trans_score(h_embed, batch_pos_t_embed, batch_pos_t_embed)
            else :
                pos_score, _ = self.calc_dot_score(h_embed, batch_pos_t_embed, batch_pos_t_embed)
            pos_score_flat = pos_score.contiguous().view(-1)
            if mask is not None :
                pos_flat_mask = mask['pos_t'].contiguous().view(-1)
                pos_score_valid = pos_score_flat[pos_flat_mask]
                return pos_score_valid
            return pos_score_flat
        
    def get_trained_entity_embed(self, test_entity, type='drug') :
        """
        Get trained entity embedding for predictions from the ID of drugs or genes
        """
        if test_entity in self.train_entity2index.keys() :
            return self.entity_embed[self.train_entity2index[test_entity]]
        else :
            gene_pretrained = self.get_pretrained_embedding()
            if self.modality == 3 :
                gene_func_pretrained_dict, gene_seq_pretrained_dict, prot_structure_embed = gene_pretrained
            elif self.modality == 2 :
                gene_func_pretrained_dict, gene_seq_pretrained_dict = gene_pretrained
            elif self.modality == 1 :
                gene_seq_pretrained_dict = gene_pretrained
            if type=='gene' :
            # Obtain the pretrained embeddings of drugs and genes
                try :
                    gene_seq_pretrained = torch.tensor(gene_seq_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                    trained_gene_seq_pretrained_dict_embed = {
                        gene : torch.tensor(gene_seq_pretrained_dict[gene]).to(self.device) for gene in self.trained_genes
                    }
                    #most_similar_drug_smiles, sim_smiles = find_most_similar_drug(drug_smiles_pretrained, trained_drug_smiles_pretrained_embed)
                    #print(543, 'model', most_similar_drug_smiles)
                    gene_seq_pretrained = self.gene_sequence_project(gene_seq_pretrained)
                    #drug_smiles = (drug_smiles_pretrained + (self.entity_embed[self.train_entity2index[most_similar_drug_smiles]])[int(self.embed_dim/self.modality):int(2 * self.embed_dim/self.modality)])/2
                    gene_seq = gene_seq_pretrained

                    if self.modality >= 2 :
                        gene_func_pretrained = torch.tensor(gene_func_pretrained_dict[test_entity], dtype=torch.float).to(self.device)
                        gene_func_pretrained = self.gene_function_project(gene_func_pretrained)  # pretrained dim -> embed dim
                        gene_func = gene_func_pretrained
                    
                    if self.modality == 3 :
                        if test_entity in prot_structure_embed :
                            prot_structure_embeddding = torch.tensor(prot_structure_embed[test_entity], dtype=torch.float).to(self.device)
                        else :
                            prot_structure_embeddding = self.gene_sequence_project(gene_seq_pretrained).to(self.device)
                    if self.modality == 3 :
                        embed = torch.cat([gene_func,gene_seq,prot_structure_embeddding], axis=-1)
                    elif self.modality == 2 :
                        embed = torch.cat([gene_func,gene_seq], axis=-1)
                    else :
                        embed = gene_seq
                    return embed
                    
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
            else :
                raise ValueError("PPI Model does not support entity of type other than genes")

                
                
        

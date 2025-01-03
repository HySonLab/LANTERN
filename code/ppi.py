import os
import torch
from utils import create_pyg_dataset, generate_bi_coo_matrix, generate_directed_coo_matrix, save_model, exclude_isolation_point,custom_collate
from parse import parse_args
from data_loader import DataProcessor, DDI_DataProcessor, PPI_DataProcessor, DrugProteinDataSet
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam, lr_scheduler, SGD
from model import Model, DDI_Model, PPI_Model
from procedure import train, test
import pickle
import random

def main() :
    args = parse_args()
    hop = args.hop
    seed = args.seed
    epoch = args.epoch
    save_path = args.save_path
    #save_model_sign = args.save_model
    save_model_sign = True
    print(21, "MAIN", save_model_sign)
    model_save_dir = os.path.split(save_path)[0]
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_dir = os.path.join(model_save_dir, str(seed))
    if not os.path.exists(model_save_dir) and save_model_sign:
        os.makedirs(model_save_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    Processing from ID to indices
    """
    train_data = PPI_DataProcessor(args.train_path, hop, mode="train")
    train_entity2index = train_data.entities2id 
    valid_data = PPI_DataProcessor(args.valid_path, hop, mode="val")
    valid_entity2index = valid_data.entities2id
    test_data = PPI_DataProcessor(args.test_path, hop, mode="test")
    test_entity2index = test_data.entities2id

    """
    Check the number of unique drugs
    """
    # Combine keys from all three dictionaries
    all_entities = set(train_entity2index.keys()).union(valid_entity2index.keys(), test_entity2index.keys())
    train_and_val = set(train_entity2index.keys()).union(valid_entity2index.keys())
    # Get the number of unique entities
    unique_entity_count = len(all_entities)

    # Print the result
    print(f"Number of unique entities across train : {len(set(train_entity2index.keys()))}")
    print(f"Number of unique entities across train and val : {len(train_and_val)}")
    print(f"Number of unique entities across all three datasets: {unique_entity_count}")

    """
    Load relations in form of indices
    """
    train_triples_data = train_data.load_data() # Positive triples represented in indices
    valid_triples_data = valid_data.load_data()
    test_triples_data = test_data.load_data()

    """
    Load related knowledge graph for message passing and update
    """
    train_kg_triples = train_triples_data # ~ others, if hop is not None, retrieve the relations related to entities in triples_data
    valid_kg_triples = valid_triples_data
    test_kg_triples = test_triples_data

    """
    Model and Graph arguments
    """
    score_fun = args.score_fun
    model_args = (args.layer_num, args.head_num)
    entity_num = train_data._calc_drug_protein_num()
    print(54, 'main', entity_num)
    #print(type(train_data.get_protein_num))
    protein_num = train_data._protein_num # Funny facts : This does not work : train_data._get_protein_num()
    drug_num = train_data._drug_num
    #relation_num = train_data.get_relation_num()
    relation_num = 2
    print(57, 'main', relation_num)
    entity_num = protein_num + drug_num
    print(61, 'main : ', f'entity_num : {entity_num}, protein_num : {protein_num}, drug_num : {drug_num}')
    
    train_set_len = len(train_triples_data)

    valid_triples_data, valid_exclusion_list = exclude_isolation_point(train_triples_data, valid_triples_data)
    train_triples_data += valid_exclusion_list

    test_triples_data, test_exclusion_list = exclude_isolation_point(train_triples_data, test_triples_data)
    train_triples_data += test_exclusion_list

    train_set = DrugProteinDataSet(train_triples_data, args.neg_ratio) # previously, train_data contains 'treats' and 'others' relation.
    print(93, 'main', train_set.len_head)
    train_data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, collate_fn=custom_collate)

    valid_set_len = len(valid_triples_data)
    valid_set = DrugProteinDataSet(valid_triples_data, args.neg_ratio)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=32, shuffle=True, collate_fn=custom_collate)

    test_set_len = len(test_triples_data)
    test_set = DrugProteinDataSet(test_triples_data, args.neg_ratio)
    test_data_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, collate_fn=custom_collate)

    drug_pretrained_dim = 768
    gene_sequence_dim = 1024
    code_folder = os.path.dirname(__file__)
    kgcnh_folder = os.path.dirname(code_folder)
    path2_pretrained_embeddings = os.path.join(os.path.dirname(kgcnh_folder), 'embeddings')
    model = PPI_Model(entity_num, drug_num, protein_num, relation_num, args.embed_dim,
                      model_args, args.enable_augmentation,
                      (args.enable_gumbel, args.tau, args.amplitude / args.epoch), args.decay, args.dropout,
                      score_fun, device, args.modality, args.dataset_name, train_entity2index, 
                      path2_pretrained_embeddings, drug_pretrained_dim, gene_sequence_dim).to(device)
    
    #optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for name, param in model.named_parameters():
        if any(param is p for p in optimizer.param_groups[0]['params']):
            print(f"{name} is in optimizer.")
    file_name = 'result' + '.pkl'
    model_save_path = os.path.join(model_save_dir, file_name)

    result = {'best_acc': 0, 'best_f1': 0,'best_auc' : 0, 'best_aupr' : 0,  'epoch' : 0, 'model' : model, 'optimizer' : optimizer}
    
    augment_graph = None
    augment_relation = None
    
    lr = args.lr
    for i in range(epoch):
        if i > 0 :
            train_set.reset_seen_heads()
        
        if i == epoch * 0.3 :
            lr = 0.8 * lr
        elif i == epoch * 0.6 :
            lr = 0.8 * lr
        elif i == epoch * 0.8 :
            lr = 0.8 * lr
        
        loss, reg_loss = train(model, train_data_loader, optimizer, device, lr, train_entity2index) # In-context learning ? Take the kg_graph (other relations related to training entities)
                                                                        # as the context for drug and protein embedping.       
        print("Line 109, FINISHED TRAINING", f"epoch {i}, loss : {loss}, reg_loss : {reg_loss}")
        if (i + 1) % args.valid_step == 0 :
            print("epoch:{},loss:{}, reg_loss:{}".format(i + 1, loss, reg_loss))
            #auc, aupr,test_acc, test_prec, test_recall, test_f1, all_score, all_label = test(model, valid_data_loader, device, valid_entity2index)
            auc, aupr, all_score, all_label = test(model, valid_data_loader, device, valid_entity2index)
            valid_set.reset_seen_heads()
            print("LINE 115, FINISHED 1 VALID")
            #print("epoch:{},auc:{}, aupr:{}".format(i + 1, auc, aupr, test_acc, test_prec, test_recall, test_f1))
            if result['best_aupr'] < aupr:
                result['epoch'] = i + 1
                result['best_auc'] = auc
                result['best_aupr'] = aupr
                #result['best_acc'] = test_acc
                #result['best_f1'] = test_f1
                result['best_score'] = all_score
                result['label'] = all_label
                result['model'] = model

                print(164, 'MAIN')
                model_arg = [entity_num, drug_num, protein_num, relation_num, args.embed_dim,
                      model_args, args.enable_augmentation,
                      (args.enable_gumbel, args.tau, args.amplitude / args.epoch), args.decay, args.dropout,
                      score_fun, device, args.modality, args.dataset_name, train_entity2index, 
                      path2_pretrained_embeddings, drug_pretrained_dim, gene_sequence_dim]
                res = {'model': model, 'model_arg' : model_arg}
                
                model_saved_path = os.path.join(kgcnh_folder, 'log', f'training', args.dataset_name, str(args.modality), f'result_{aupr}_{i}.pkl')
                if save_model_sign and aupr > 0.915 :
                    save_model(res, model_saved_path)
    save_model(result, save_path)
    print(result['epoch'], result['best_auc'], result['best_aupr'])
    print("model_save_path:", save_path)
    auc, aupr, all_score, all_label = test(result['model'], test_data_loader, device, test_entity2index)
    print('\n', 'TEST : ', auc, aupr)

if __name__ == '__main__':
    main()
    
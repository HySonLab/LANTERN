import os
import torch
from utils import save_model,custom_collate
import argparse
from data_loader import DataProcessor, DDI_DataProcessor, PPI_DataProcessor, DrugProteinDataSet, ProteinProteinDataSet, DrugDrugDataSet
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam, lr_scheduler, SGD
from model import Model, DDI_Model, PPI_Model
from procedure import train, test
import pickle
import random


code_path = os.path.dirname((__file__))
lantern_path = os.path.dirname(code_path)
print(lantern_path)
biosnap_path = os.path.join(lantern_path, 'data', 'BioSNAP')
davis_path = os.path.join(lantern_path, 'data', 'DAVIS')
kiba_path = os.path.join(lantern_path, 'data', 'KIBA')
yeast_path = os.path.join(lantern_path, 'data', 'yeast')
bindingdb_path = os.path.join(lantern_path, 'data', 'BindingDB')

def main(args) :
    seed = args.seed
    epoch = args.epoch
    save_path = args.save_path
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
    interaction_type = args.interaction_type
    print(interaction_type)
    data_path = os.path.join(lantern_path, 'data', args.dataset_name)
    print(data_path)
    if interaction_type == 'PPI' :
      train_data = PPI_DataProcessor(data_path, mode="train")
      valid_data = PPI_DataProcessor(data_path,mode="val")
      test_data = PPI_DataProcessor(data_path, mode="test")
    elif interaction_type == 'DDI' :
      train_data = DDI_DataProcessor(data_path, mode="train")
      valid_data = DDI_DataProcessor(data_path,mode="val")
      test_data = DDI_DataProcessor(data_path, mode="test")
    elif interaction_type == 'DTI' :
      train_data = DataProcessor(data_path, mode="train")
      valid_data = DataProcessor(data_path,mode="val")
      test_data = DataProcessor(data_path, mode="test")
    train_entity2index = train_data.entities2id
    valid_entity2index = valid_data.entities2id
    test_entity2index = test_data.entities2id

    """
    Check the number of unique drugs
    """
    # Combine keys from all three dictionaries
    all_entities = set(train_entity2index.keys()).union(valid_entity2index.keys(), test_entity2index.keys())
    train_and_val = set(train_entity2index.keys()).union(valid_entity2index.keys())
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
    Model arguments
    """
    score_fun = args.score_fun
    entity_num = train_data._calc_drug_protein_num()
    print(54, 'main', entity_num)
    protein_num = train_data._protein_num
    drug_num = train_data._drug_num
    entity_num = protein_num + drug_num
    print(61, 'main : ', f'entity_num : {entity_num}, protein_num : {protein_num}, drug_num : {drug_num}')
    
    train_set_len = len(train_triples_data)
    if interaction_type == 'PPI' :
      train_set = ProteinProteinDataSet(train_triples_data, mode='train') # previously, train_data contains 'treats' and 'others' relation.
      valid_set = ProteinProteinDataSet(valid_triples_data, mode='val')
      test_set = ProteinProteinDataSet(test_triples_data, mode='test')
    elif interaction_type == 'DDI' :
      train_set = DrugDrugDataSet(train_triples_data, mode='train') # previously, train_data contains 'treats' and 'others' relation.
      valid_set = DrugDrugDataSet(valid_triples_data, mode='val')
      test_set = DrugDrugDataSet(test_triples_data, mode='test')
    elif interaction_type == 'DTI' :
      train_set = DrugProteinDataSet(train_triples_data, mode='train') # previously, train_data contains 'treats' and 'others' relation.
      valid_set = DrugProteinDataSet(valid_triples_data, mode='val')
      test_set = DrugProteinDataSet(test_triples_data, mode='test')
    train_data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, collate_fn=custom_collate)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=32, shuffle=True, collate_fn=custom_collate)
    test_data_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, collate_fn=custom_collate)

    drug_pretrained_dim = 768
    gene_sequence_dim = 1024
    code_folder = os.path.dirname(__file__)
    lantern_folder = os.path.dirname(code_folder)
    print(114, lantern_path)
    path2_pretrained_embeddings = os.path.join(lantern_folder, 'data', args.dataset_name)
    if interaction_type == 'PPI' :
      model = PPI_Model(entity_num, drug_num, protein_num, args.embed_dim, args.decay, args.dropout,
                        score_fun, device, args.modality, args.dataset_name, train_entity2index, 
                        path2_pretrained_embeddings, drug_pretrained_dim, gene_sequence_dim).to(device)
    elif interaction_type == 'DDI' :
      model = DDI_Model(entity_num, drug_num, protein_num, args.embed_dim, args.decay, args.dropout,
                        score_fun, device, args.modality, args.dataset_name, train_entity2index, 
                        path2_pretrained_embeddings, drug_pretrained_dim, gene_sequence_dim).to(device)
    elif interaction_type == 'DTI' :
      model = Model(entity_num, drug_num, protein_num, args.embed_dim, args.decay, args.dropout,
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
            auc, aupr, other_metrics, all_score, all_label = test(model, valid_data_loader, device, valid_entity2index)
            print(f"Valid score epoch {i+1} : auc : {auc} aupr : {aupr} other_metrics : {other_metrics}")
            valid_set.reset_seen_heads()
            print("LINE 115, FINISHED 1 VALID")
            if result['best_aupr'] < aupr:
                result['epoch'] = i + 1
                result['best_auc'] = auc
                result['best_aupr'] = aupr
                result['best_score'] = all_score
                result['label'] = all_label
                result['model'] = model

                print(164, 'MAIN')
                model_arg = [entity_num, drug_num, protein_num, args.embed_dim, args.decay, args.dropout,
                      score_fun, device, args.modality, args.dataset_name, train_entity2index, 
                      path2_pretrained_embeddings, drug_pretrained_dim, gene_sequence_dim]
                res = {'model': model, 'model_arg' : model_arg}
                
                model_saved_path = os.path.join(lantern_folder, 'log', f'training', args.dataset_name, str(args.modality), f'result_{aupr}_{i}.pkl')
                if save_model_sign :
                    save_model(res, model_saved_path)
    save_model(result, save_path)
    print(result['epoch'], result['best_auc'], result['best_aupr'])
    print("model_save_path:", save_path)
    auc, aupr, all_score, all_label = test(result['model'], test_data_loader, device, test_entity2index)
    print('\n', 'TEST : ', auc, aupr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="456")
    parser.add_argument('--interaction_type', type=str, default='PPI')
    parser.add_argument('--dataset_name', type=str, default="yeast")
    parser.add_argument('--embed_dim', type=int, default=384,
                        help="the embedding size entity and relation")
    parser.add_argument('--decay', type=float, default=1e-6,
                        help="the weight decay for l2 regulation")
    parser.add_argument('--seed', type=int, default=120) # 42, 85, 100
    parser.add_argument('--valid_step', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="the learning rate")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="using the dropout ratio")
    parser.add_argument('--num_folds', type=float, default=1)
    parser.add_argument('--modality', type=int, default=1)
    parser.add_argument('--save_model', default = True, action='store_true', help='save_model')
    parser.add_argument('--save_path', nargs='?', default=os.path.join(lantern_path, 'log' , 'result.pkl'), help='Input save path.')
    parser.add_argument('--score_fun', nargs='?', default='transformer', help='Input data path.')
    parser.add_argument('--drug_pretrained_dim', type=int, default = 768)
    parser.add_argument('--protein_sequence_dim', type=int, default = 1024)
    args = parser.parse_args()
    main(args)
      
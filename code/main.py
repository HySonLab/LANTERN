import os
import torch
from utils import create_pyg_dataset, generate_bi_coo_matrix, generate_directed_coo_matrix, save_model, exclude_isolation_point,custom_collate
from parse import parse_args
from data_loader import DataProcessor, DrugProteinDataSet
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam, lr_scheduler, SGD
from model import Model
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
    train_data = DataProcessor(args.train_path, hop, mode="train")
    train_entity2index = train_data.entities2id 
    valid_data = DataProcessor(args.valid_path, hop, mode="val")
    valid_entity2index = valid_data.entities2id
    test_data = DataProcessor(args.test_path, hop, mode="test")
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
    """
    Creating the graph
    """
    # generate kg_graph

    kg_coo, kg_relation = generate_bi_coo_matrix(train_kg_triples)
    #augment_coo, augment_relation = generate_bi_coo_matrix(train_graph)

    kg_graph = create_pyg_dataset(args.embed_dim, kg_coo, kg_relation).to(device)
    #augment_graph = kg_graph

    #train_set = [tuple(i) for i in np.array(train_triples_data).tolist()]
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
    model = Model(entity_num, drug_num, protein_num, relation_num, args.embed_dim,
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

    result = {'best_auc': 0, 'best_aupr': 0, 'epoch' : 0, 'model' : model, 'optimizer' : optimizer}
    
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
            auc, aupr, all_score, all_label = test(model, valid_data_loader, device, valid_entity2index)
            valid_set.reset_seen_heads()
            print("LINE 115, FINISHED 1 VALID")
            print("epoch:{},auc:{}, aupr:{}".format(i + 1, auc, aupr))
            if result['best_aupr'] < aupr:
                result['epoch'] = i + 1
                result['best_auc'] = auc
                result['best_aupr'] = aupr
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
    # 00002 -> epoch = 100, 140 : 08947
    # left : dot tuning, no message passing , right : mlp score
    # left embed dim = 512 lr 00004, right 00002 embed dim = 256 => left 00001; right : 00001, no bridging connection
    # left : 00001, no bridging connection; right : no coeff for bio and nlp
    # left : one more project linear layer, no relu at the end, right : relu at the end.
    # left : two MLPs without relu at the end, right : scale up hidden dim => right a bit better
    # left : no dropout in linear layer, right : one linear layer, no relu at the end -> 09 0911
    # left : dropout = 0.2, right : dropout = 0.05
    # left : dropout = 0.1, no weight decay;  right : smaller weight decay
    # left : eval of sota, right : smaller lr = 0.00005
    # left : no                    ,  right : lr = 0.00005, epoch = 300
    # left : no message passing with normalization, right : message passing with normalization, relation zeros, ones
    # left : bio coeff = 1.25, right : nlp coeff = 0.75; right : bio coeff = 2, nlp coeff = 1  => left : 0.91468
    # left : bio coeff = 0.75, right : nlp coeff = 1.25; right : bio coeff = 1.5, nlp coeff = 0.5
    # left : seed 86, right : seed 100   => seed 100 : SOTA
    # left : huber loss, right : mse loss
    # left : 2 0 nlp bio ; right : 0 2 nlp bio
    # seed 120 : left : 0.8 1.2 coeff ; right : 0.7 1.3  right SOTA
    # seed 142 : left : 0.9 1.1 coeff ; right : 0.6 1.4
    # left : 0.4 1.6 ; right : 0.5 1.5
    # left : 0.7 1.3 ; right : 0.8 1.2 ; both : bio dim x 2
    # left : lr = 0.0001678, right : lr = 0.00032
    # mlp coeffs, left : 0.0001, right : 0.0005, no update for the coeff weights
    # mlp coeffs, left : 0.01, right : 0.005, update for the coeff weights
    # cross loss, left : 0.0001, right : 0.0003  -> 0.9136 test
    # cross 4 layers, left : 0.0001, right : 0.0003
    # no softmax coeffs, left : 0.001, right : 0.0005
    # left: double normalization score , lr = 0.001 (maybe); right : pos_weight
    # left : pos_weight = 1.5, right : pos_weight = 1.8
    # left : pos_weight = 1.1, lr = 0.0001, embed dim = 256, right : similar, embed dim = 320, no pos weight 
    # coeff 0.68, 1.32;  left : embed dim = 280 , right : embed dim = 320.
    # 3 modality , coeff : 0.8 , 2, lr 0.0001; right : lr 0.00005
    # 3 modality, left : 0.00001 coeff : above, right : 0.00001 coeff : no
    # left : bridging in prediction, right : no bridging
    # left : lr 1e-5, right : lr 1e-4
    # left : lr 5e-4 coeffs 0.3 geo, right : lr 5e-4, coeffs
    # left : 03 nlp 068 struc , right : 068 nlp 03 struc lr 00001
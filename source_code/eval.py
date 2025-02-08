import argparse
import torch
import os
import pickle
import sys
import re
from procedure import test
from utils import custom_collate
from data_loader import DataProcessor,DDI_DataProcessor,PPI_DataProcessor,DrugProteinDataSet,DrugDrugDataSet,ProteinProteinDataSet
from torch.utils.data import DataLoader
import numpy as np
import random
import io

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Adjust sys.path to ensure modules are found

code_folder = os.path.dirname(__file__)
lantern_folder = os.path.dirname(code_folder)
sys.path.append(lantern_folder)

biosnap_path = os.path.join(lantern_folder, 'data', 'BioSNAP')
davis_path = os.path.join(lantern_folder, 'data', 'DAVIS')
deepddi_path = os.path.join(lantern_folder, 'data', 'DeepDDI')
yeast_path = os.path.join(lantern_folder, 'data', 'yeast')
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "model":
            module = "source_code.model"
        # Special case for torch.storage._load_from_bytes
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)
    
def evaluate(args) :
    if torch.cuda.is_available() and args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    seed = 24 #
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(args.model_save_path, "rb") as f:
      res = CustomUnpickler(f).load()
    print("Type res :", type(res))
    print("Keys of res :" , res.keys())
    model = res['model']
    model.device = device
    #model.path_2_pretrained_embedding = os.path.join(lantern_folder, 'data', args.dataset_name)
    model.eval()
    interaction_type = args.interaction_type
    dataset_name = args.dataset_name
    test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', dataset_name)
    if interaction_type == 'PPI' :
      test_data = PPI_DataProcessor(test_path, mode="test")
    elif interaction_type == 'DDI' :
      test_data = DDI_DataProcessor(test_path, mode="test")
    elif interaction_type == 'DTI' :
      test_data = DataProcessor(test_path, mode="test")
    test_entity2index = test_data.entities2id
    test_triples_data = test_data.load_data()
    test_set_len = len(test_triples_data)
    if interaction_type == 'PPI' :
      test_set = ProteinProteinDataSet(test_triples_data, mode='eval')
    elif interaction_type == 'DDI' :
      test_set = DrugDrugDataSet(test_triples_data, mode='eval')
    elif interaction_type == 'DTI' :
      test_set = DrugProteinDataSet(test_triples_data, mode='eval')
    test_data_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, collate_fn=custom_collate)
    auc, aupr, other_metrics, all_score, all_label = test(model, test_data_loader, device, test_entity2index)
    print("auc, aupr :", auc, aupr, other_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="add args for predictions")
    parser.add_argument("--model_save_path", default=os.path.join(lantern_folder, 'log','training','yeast', '1', 'result_0.9603380175462447_29.pkl'), help="path to model saved")
    parser.add_argument("--interaction_type", type=str, default='DTI')
    parser.add_argument("--dataset_name", type=str, default='yeast')
    #parser.add_argument("--model_save_path", default=os.path.join(lantern_folder, 'log','training','BioSNAP', '1', 'result_0.992936974798785_19.pkl'), help="path to model saved")
    #parser.add_argument('--test_path', default=yeast_path)
    parser.add_argument("--gpu", default=False, help="enable gpu")
    args = parser.parse_args()
    evaluate(args)
    
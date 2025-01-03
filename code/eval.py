import argparse
import torch
import os
import pickle
import sys
import re
from procedure import test
from utils import custom_collate
from data_loader import DataProcessor, DrugProteinDataSet, PPI_DataProcessor
from torch.utils.data import DataLoader
import numpy as np
import random
import io

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adjust sys.path to ensure modules are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
code_folder = os.path.dirname(__file__)
kgcnh_folder = os.path.dirname(code_folder)
kgdrp_folder = os.path.dirname(kgcnh_folder)

kgcnh_path = os.path.dirname(os.path.dirname(__file__))
biosnap_path = os.path.join(kgcnh_path, 'data', 'BioSNAP')
davis_path = os.path.join(kgcnh_path, 'data', 'DAVIS')
deepddi_path = os.path.join(kgcnh_path, 'data', 'DeepDDI')
yeast_path = os.path.join(kgcnh_path, 'data', 'yeast')
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "model":
            module = "KGCNH.code.model"
        elif module == "layers":
            module = "KGCNH.code.layers"
        # Special case for torch.storage._load_from_bytes
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)
    
def evaluate(args) :
    if torch.cuda.is_available() and args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    seed = 24 # 120 auc : 0.963, aupr : 0.949; 10 auc : 0.955, aupr : 0.937; 1200 : 0.953, 0.943, result_aupr = 0.99135, num_head = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(args.model_save_path, "rb") as f:
      res = CustomUnpickler(f).load()
    print("Type res :", type(res))
    print("Keys of res :" , res.keys())
    model = res['model']
    model.device = device
    #model.dataset_name = 'DAVIS'
    #model.path_2_pretrained_embedding = os.path.join(os.path.dirname(kgcnh_folder), 'embeddings')
    #model.modality = 1
    model.eval()
    hop = args.hop
    
    #test_data = PPI_DataProcessor(args.test_path, hop, mode="test")
    test_data = DataProcessor(args.test_path, hop, mode="test")
    test_entity2index = test_data.entities2id
    test_triples_data = test_data.load_data()
    test_set_len = len(test_triples_data)
    test_set = DrugProteinDataSet(test_triples_data, args.neg_ratio, mode='eval')
    test_data_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, collate_fn=custom_collate)
    
    #auc, aupr,test_acc, test_prec, test_recall, test_f1, all_score, all_label = test(model, test_data_loader, device, test_entity2index)
    auc, aupr, all_score, all_label = test(model, test_data_loader, device, test_entity2index)
    print("auc, aupr :", auc, aupr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="add args for predictions")
    #parser.add_argument("--model_save_path", default=os.path.join(kgcnh_folder, 'log','training','yeast', '1', 'result_0.9926545768799635_29.pkl'), help="path to model saved")
    parser.add_argument("--model_save_path", default=os.path.join(kgcnh_folder, 'log','training','BioSNAP', '1', 'result_0.992936974798785_19.pkl'), help="path to model saved")
    parser.add_argument('--test_path', default=biosnap_path)
    parser.add_argument("--gpu", default=False, help="enable gpu")
    parser.add_argument("--model_args", default='log/new/model_arg.pkl')
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--neg_ratio', type=int, default=10)
    #parser.add_argument('--data_processor_mode', type=str, default='test')
    args = parser.parse_args()
    evaluate(args)
    # unseen genes : auc, aupr : 0.7115384615384616 0.375765931372549
    # left : bs : 32, right : bs : 64  -> left : full gene similarity based
    # left : 0.916 , right : 0.9154  -> auc, aupr : 0.8965576644100579 0.9125145757193357 and auc, aupr : 0.8969965634424582 0.9127983521133411
    # left : 0.917 , right : 0.918  -> 0.8959690219807225 0.912772793007035 0.8987202299308429 0.9140577942797397
    # left : 0.919, right : 0.9182 -> 0.8964415261380958 0.9108001651945754, 0.8961441514607085 0.9132915502248483
    # left : 0.9154, right : 0.9156  
    # left : 0.9160, right : 0.9162 -> mlpcoeffs: auc, aupr : 0.899002434309335 0.9131285382790445 auc, aupr : 0.898750657946665 0.9136332360843902.
    # left : modal2 auc, aupr : 0.9998420315023002 0.9994606583382408, right : modal1 auc, aupr : 0.9992060445419371 0.9987153093758113
    # DAVIS : test : 0.9878201813552735 0.8495056967042243
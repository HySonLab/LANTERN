import argparse
import os

kgcnh_path = os.path.dirname(os.path.dirname(__file__))
kgdrp = os.path.dirname(kgcnh_path)
biosnap_path = os.path.join(kgcnh_path, 'data', 'BioSNAP')
davis_path = os.path.join(kgcnh_path, 'data', 'DAVIS')
kiba_path = os.path.join(kgcnh_path, 'data', 'KIBA')
yeast_path = os.path.join(kgcnh_path, 'data', 'yeast')
bindingdb_path = os.path.join(kgcnh_path, 'data', 'BindingDB')

def parse_args():
    parser = argparse.ArgumentParser(description="456")
    parser.add_argument('--gpu', action='store_true', help='enable gpu')
    parser.add_argument('--save_model', action='store_true', help='save_model')

    parser.add_argument('--enable_gumbel', action='store_true', help='enable gumbel-softmax')
    parser.add_argument('--enable_augmentation', action='store_true', help='enable_augmentation')
    parser.add_argument('--save_path', nargs='?', default=os.path.join(kgdrp, 'log' , 'result.pkl'), help='Input save path.')
    parser.add_argument('--data_path', nargs='?', default='./data/Hetionet',
                        help='Input data path.')
    #parser.add_argument('--score_fun', nargs='?', default='dot', help='Input data path.')
    parser.add_argument('--score_fun', nargs='?', default='mlp', help='Input data path.')
    parser.add_argument('--embed_dim', type=int, default=384,
                        help="the embedding size entity and relation")
    parser.add_argument('--seed', type=int, default=120) # 42, 85, 100
    parser.add_argument('--valid_step', type=int, default=10)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--layer_num', type=int, default=3,
                        help="the layer num")
    parser.add_argument('--neg_ratio', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="the learning rate")
    parser.add_argument('--tau', type=float, default=1.3,
                        help="the learning rate")
    parser.add_argument('--amplitude', type=float, default=0.6,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-6,
                        help="the weight decay for l2 regulation")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="using the dropout ratio")
    parser.add_argument('--head_num', type=int, default=2,
                        help="the head num")
    parser.add_argument('--modality', type=int, default=1)
    #parser.add_argument('--dataset_name', type=str, default="BioSNAP")
    #parser.add_argument('--train_path', default=biosnap_path)
    #parser.add_argument('--valid_path', default=biosnap_path)
    #parser.add_argument('--test_path', default=biosnap_path)
    parser.add_argument('--dataset_name', type=str, default="yeast")
    parser.add_argument('--train_path', default=yeast_path)
    parser.add_argument('--valid_path', default=yeast_path)
    parser.add_argument('--test_path', default=yeast_path)
    return parser.parse_args()

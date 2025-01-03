import functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import pytorch_lightning as pl
import torch
import torch_geometric
from torch_geometric.graphgym.models import AtomEncoder, BondEncoder
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import numpy as np
#from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch
import torch.utils.data as data
import torch_cluster
import math
import os 
import atom3d.util.formats as fo 
import json
from collections import defaultdict
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


code_folder = os.path.dirname(__file__)
kgcnh_folder = os.path.dirname(code_folder)
kgdrp_folder = os.path.dirname(kgcnh_folder)
genes2seq_dict_path = os.path.join(kgdrp_folder, 'embeddings', 'pretrained', 'gene', 'gene_sequence.json')
prot2structure_path = os.path.join(kgdrp_folder, 'embeddings', 'pretrained', 'gene', 'prot_structure.json')
outlier_prot_path = 'outlier_prot.txt'

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x
    

_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
def get_dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def positional_embeddings(edge_index, 
                            num_embeddings=16,
                            period_range=[2, 1000], device ="cuda"):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device = device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def get_sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 

def featurize_as_graph(protein, num_rbf = 16, device = "cuda"):
    try :
        name = protein['name']
        N_coords = protein[protein.name == 'N'][['x', 'y', 'z']].to_numpy()
        CA_coords =protein[protein.name == 'CA'][['x', 'y', 'z']].to_numpy()
        C_coords = protein[protein.name == 'C'][['x', 'y', 'z']].to_numpy()
        O_coords = protein[protein.name == 'O'][['x', 'y', 'z']].to_numpy()
        max_ = min([CA_coords.shape[0], C_coords.shape[0], N_coords.shape[0], O_coords.shape[0]])
        #max_ = CA_coords.shape[0]
        N_coords = N_coords[:max_, :]
        CA_coords = CA_coords[:max_, :]
        C_coords = C_coords[:max_, :]
        O_coords = O_coords[:max_, :]
        coords = np.stack((N_coords, CA_coords,C_coords, O_coords), axis = 1)
        #print(coords.shape)
        with torch.no_grad():
            coords = torch.from_numpy(coords).float().to(device)
            seq = torch.as_tensor([_amino_acids(a) for a in protein[protein.name == 'CA']['resname'][:max_]],
                                    dtype=torch.long).to(device)
            # print([a for a in protein[protein.name == "CA"]['resname'][:max_] if _amino_acids(a) == 20])
            # exit(0)
            #print(coords.shape)
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            #print(X_ca.shape)
            #X_ca = coords
            edge_index = torch_cluster.radius_graph(X_ca, r = 5.0)
            #edge_index = torch_cluster.knn_graph(X_ca, k=30)
            
            pos_embeddings = positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device = device)
            
            dihedrals = get_dihedrals(coords)                     
            orientations = get_orientations(X_ca)
            sidechains = get_sidechains(coords)
            
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
        data = torch_geometric.data.Data(x=X_ca, seq = seq,
                                            node_s=node_s, node_v=node_v,
                                            edge_s=edge_s, edge_v=edge_v,
                                            edge_index=edge_index) #mask=mask)
        return data
    except Exception as e :
        import traceback
        print("An error occurred:", str(e))
        traceback.print_exc()  # This will print the stack trace of the exception
    
                            
class ThreeD_Protein_Model(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.5, attention_type = "performer"):
        
        super().__init__()
         
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), vector_gate=True))
            
        self.attention_type = attention_type
        layer = nn.TransformerEncoderLayer(ns, 4, ns * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
 
        self.skip_connection = nn.Sequential(nn.Linear(ns * 2, ns), nn.ReLU(), nn.Linear(ns, ns))
         
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        ''' 
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V) 
        x, mask = to_dense_batch(out, batch) 
        x_o = self.transformer(x)
        x = torch.cat([x, x_o], dim = -1)
        x = self.skip_connection(x)
        geo_rep = x.mean(dim = 1)
        if seq is not None:
            z = torch.cat([geo_rep, seq], dim = -1)
            return z
        return geo_rep

class Protein3DRepresentation(nn.Module):
    def __init__(self, hidden_dim = 128, attention_type = "transformer"):
        super().__init__()
        self.protein_model = ThreeD_Protein_Model(node_in_dim = (6,3), node_h_dim = (hidden_dim, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                      num_layers = 3, drop_rate=0.1, attention_type = attention_type)
    def forward(self, protein_graph):
        x_prot = self.protein_model((protein_graph.node_s, protein_graph.node_v), 
                                   protein_graph.edge_index, (protein_graph.edge_s, protein_graph.edge_v), seq = None, batch = protein_graph.batch)
        return x_prot

 
def create_protein_graph(uniprot_id) :
    """
    Construct the protein graph from the protein structure.
    """
    root = "/home/stuhcnga/scratch/KGDRP/protein_3d/"
    pdb_path = os.path.join(root, f'{uniprot_id}.pdb')
    protein_df = fo.bp_to_df(fo.read_pdb(pdb_path))
    #print("632, protein_df : ", protein_df)
    protein_graph = featurize_as_graph(protein_df)
    #print("634, protein graph :", protein_graph)
    return protein_graph

#if __name__=='__main__' :
def get_3d_pro_representation(uniprot_id) :
    #uniprot_id = 'Q5T4S7'
    protein_graph = create_protein_graph(uniprot_id)
    protein_graph = protein_graph.to(device)
    model = Protein3DRepresentation()
    model.eval()
    model = model.to(device)
    with torch.no_grad() :
        ThreeD_representation = model(protein_graph)
    print(ThreeD_representation)
    print(ThreeD_representation.shape)
    return ThreeD_representation

def save_3d_representation(gene2sequence_path) :
    model = Protein3DRepresentation()
    model.eval()
    model = model.to(device)
    with open(gene2sequence_path, 'r') as f :
        gene_dict = json.load(f)
    #os.makedirs(prot2structure_path, exist_ok=True)
    prot2struc = defaultdict(list)
    outlier_prots = []
    for gene in gene_dict :
        try :
            protein_graph = create_protein_graph(gene)
            protein_graph = protein_graph.to(device)
            with torch.no_grad() :
                ThreeD_representation = model(protein_graph)
            if ThreeD_representation is not None :
                prot2struc[gene] = ThreeD_representation.view(-1).tolist()
            else :
                outlier_prots.append(gene)
        except Exception as e :
            print(676, e, traceback.format_exc())
        #break
    with open(prot2structure_path, 'w') as f :
        json.dump(prot2struc, f, indent=4)
        print(f'Saved protein structure to {prot2structure_path}')
    with open(outlier_prot_path, 'w') as f :
        f.write('\n'.join(outlier_prots))
        print(f"Saved outlier proteins to {outlier_prot_path}")
    
#save_3d_representation(genes2seq_dict_path)
#get_3d_pro_representation('Q5T4S7')
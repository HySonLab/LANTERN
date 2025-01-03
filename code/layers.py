import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from torch_geometric.utils import softmax

def gumbel_softmax_sample(logits, tau=1.0):
    # Draw sample from Gumbel distribution
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


class KGConv(MessagePassing):
    def __init__(self, in_channels):
        super(KGConv, self).__init__(aggr='add')  # "Add" aggregation

        # Linear transformation for node and edge features
        self.lin = nn.Linear(in_channels, in_channels)

        # Attention mechanisms
        self.att_i = nn.Linear(in_channels, 1)  # For source node attention (i.e., node sending message)
        self.att_j = nn.Linear(in_channels, 1)  # For destination node attention (i.e., node receiving message)
        
        # Optional: learnable edge weight
        self.edge_weight = nn.Linear(in_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, in_channels).
            edge_index (Tensor): Edge indices of shape (2, num_edges).
            edge_attr (Tensor): Edge features of shape (num_edges, in_channels).
        Returns:
            Tensor: Node embeddings after GAT-based message passing.
        """
        # Step 1: Add self-loops to the adjacency matrix
        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Step 2: Linearly transform node feature matrix
        x = self.lin(x)

        # Step 3: Compute attention coefficients and propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        """
        The message function computes messages sent from source nodes to target nodes, incorporating attention.
        
        Args:
            x_i (Tensor): Target node features.
            x_j (Tensor): Source node features.
            edge_attr (Tensor): Edge features.
            edge_index (Tensor): Edge indices.
            size (int): Number of nodes.
        
        Returns:
            Tensor: Messages to be aggregated.
        """
        # Compute attention coefficients for the source and target nodes
        alpha_i = self.att_i(x_i)  # Attention score for the target node
        alpha_j = self.att_j(x_j)  # Attention score for the source node

        # Combine the attention scores with the edge features
        alpha = F.leaky_relu(alpha_i + alpha_j + self.edge_weight(edge_attr))

        # Normalize attention coefficients using softmax
        alpha = softmax(alpha, edge_index[0], num_nodes=size[0])

        # Message is weighted by the attention coefficients
        return alpha * (x_j + edge_attr)

    def update(self, aggr_out):
        """
        The update function is called after aggregation to update the node embeddings.
        
        Args:
            aggr_out (Tensor): Aggregated messages for each node.
        
        Returns:
            Tensor: Updated node embeddings.
        """
        # Apply ReLU after aggregation for non-negative embeddings
        return F.relu(aggr_out)

class MKGCNH(nn.Module):
    def __init__(self, in_channels, relation_dim):
        super(MKGCNH, self).__init__()
        self.conv1 = KGConv(in_channels)
        self.conv2 = KGConv(in_channels)
        self.relation_lin = nn.Linear(relation_dim, in_channels)

    def forward(self, g, feat, r):
        # Pass relation embeddings through a linear layer
        r = self.relation_lin(r)

        g.edge_index = g.edge_index.to(feat.device)
        r = r.to(feat.device)
        # Apply the first GNN layer
        x = self.conv1(feat, g.edge_index, r) # 
        #res_x = x
        # Apply a non-linearity
        x = F.relu(x)

        # Apply the second GNN layer
        x = self.conv2(x, g.edge_index, r) # , r
        #x = x + res_x

        # Ensure the output shape matches the input shape
        return x

class Normal(nn.Module):
    def __init__(self, feature_shape, eps=1e-10):
        super(Normal, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_shape))
        self.bias = nn.Parameter(torch.zeros(feature_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / torch.pow((std + self.eps), 0.5) + self.bias
    
class KGCNH(nn.Module):
    def __init__(self, feature_shape, num_heads, p_drop, enable_contrastive, gumbel_args):
        super(KGCNH, self).__init__()
        self.mkgcnh = MKGCNH(feature_shape[-1], feature_shape[-1])
        self.drop = nn.Dropout(p_drop)
        self.normal = Normal(feature_shape)
        self.mkgcnh_bio = MKGCNH(int(feature_shape[-1]/2), feature_shape[-1])
        self.normal_specialized = Normal((feature_shape[0], feature_shape[-1]//2))
        self.mkgcnh_nlp = MKGCNH(int(feature_shape[-1]/2), feature_shape[-1])

    def forward(self, g, feat, r):
        #print(131, "layers")
        
        # Convert edge_index to tensor if necessary
        if isinstance(g.edge_index, list):
            g.edge_index = torch.tensor(g.edge_index, dtype=torch.long)
        
        bio_feat = self.mkgcnh_bio(g, feat[:, int(feat.shape[1]/2) :], r)
        bio_feat = self.normal_specialized(self.drop(bio_feat))
        bio_feat = feat[:, int(feat.shape[1]/2) :] + bio_feat # bio_feat residual connections

        nlp_feat = self.mkgcnh_nlp(g, feat[:, :int(feat.shape[1]/2)], r)
        nlp_feat = self.normal_specialized(self.drop(nlp_feat))
        nlp_feat = feat[:, :int(feat.shape[1]/2)] + nlp_feat # bio_feat residual connections

        specialized_feat = torch.cat([nlp_feat, bio_feat], dim=-1)
        # Compute the convolutional features
        conv_feat = self.mkgcnh(g, feat, r)
        conv_feat = self.normal(self.drop(conv_feat))
        # Residual connection
        res_feat = feat + conv_feat
        #res_feat = self.normal(res_feat + specialized_feat)
        res_feat = self.normal(specialized_feat)
        #res_feat = self.normal(res_feat)
        #print(153, 'layers')
        return res_feat
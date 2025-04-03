import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.module import Module


# Graphsage layer
class SageLayer(Module):
    def __init__(self, input_dim, output_dim):
        super(SageLayer, self).__init__()

        self.linear = nn.Linear(input_dim * 2, output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the weights of the linear layer with normal distribution
        nn.init.normal_(self.linear.weight)

        # Set the biases of the linear layer to a constant value of 0
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)

    def forward(self, features, adj):
        neigh_feature = adj @ features 
        
        if adj.layout != torch.sparse_coo:
            neigh_num = adj.sum(dim=1).reshape(adj.shape[0], -1) + 1
        else:
            neigh_num = adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1
        
        neigh_feature = neigh_feature / neigh_num
        
        combined = torch.cat((features, neigh_feature), dim=1)
        combined = self.linear(combined)
        output = F.relu(combined)

        return output


# transform (features, adj) to embedding vectors, through one sageLayer
class SageEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super(SageEncoder, self).__init__()

        self.sage = SageLayer(input_dim, embed_dim)
        self.dropout = dropout

    def forward(self, features, adj):
        x = self.sage(features, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class SageClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, class_num, dropout):
        super(SageClassifier, self).__init__()

        self.encoder = SageEncoder(input_dim, embed_dim, dropout)
        self.linear = nn.Linear(embed_dim, class_num)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)

    def forward(self, node_features, adjacency_list):
        x = self.encoder(node_features, adjacency_list)
        y = self.linear(x)

        return y

# link prediction on node embeddings
class GraphDecoder(nn.Module):
    def __init__(self, embed_dim):
        super(GraphDecoder, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings):
        # Linear transformation of node embeddings
        combine = F.linear(embeddings, self.weight)
        # Compute adjacency matrix using the combination of embeddings
        adj = torch.sigmoid(torch.mm(combine, combine.t()))

        return adj

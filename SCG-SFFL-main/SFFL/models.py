import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from gensim.models import Word2Vec
import torch.nn.init as init
import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '1'
      
class PositionEncoder(nn.Module):
    def __init__(self, sample_num, hidden_dim):
        super(PositionEncoder, self).__init__()
        self.linear = nn.Linear(sample_num, hidden_dim)
        init.xavier_uniform_(self.linear.weight)
        self.sample_num = sample_num
    
    def forward(self):
        with torch.no_grad():
            # Generate one-hot vectors for position encoding
            one_hot = torch.eye(self.sample_num)
            embeddings = self.linear(one_hot)

        return embeddings

class SemanticEncoder(nn.Module):
    def __init__(self, vocab, hidden_dim, epoch, random_seed=1):
        super(SemanticEncoder, self).__init__()

        self.epoch = epoch
        self.word2vec = Word2Vec(sentences=vocab, vector_size=hidden_dim, window=5, min_count=1, workers=8, seed=random_seed)

    def forward(self, tokens_list):
        with torch.no_grad():
            # Train Word2Vec model on tokens_list
            self.word2vec.train(tokens_list, total_examples=self.word2vec.corpus_count, epochs=self.epoch)
            
            # Calculate embeddings by averaging word vectors
            embeddings = [np.mean([self.word2vec.wv[token] for token in tokens if token in self.word2vec.wv], axis=0) for tokens in tokens_list]
        
        return torch.tensor(embeddings)


class GNNReconstructor(nn.Module):
    def __init__(self, hidden_dim=128, conv="GAT", head_num=8, aggr="mean", dropout=0.1):
        super(GNNReconstructor, self).__init__()

        self.dropout = dropout

        if conv == "GAT":
            self.conv1 = GATConv(hidden_dim, int(hidden_dim/head_num), heads=head_num)
            self.conv2 = GATConv(hidden_dim, int(hidden_dim/head_num), heads=head_num)
            self.conv3 = GATConv(hidden_dim, int(hidden_dim/head_num), heads=head_num)
        
        elif conv == "GCN":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        elif conv == "Sage":
            self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)

    def forward(self, mc_own_adj, mc_call_adj, m_features, c_features):

        m_features = F.dropout(m_features, p=self.dropout)
        c_features = F.dropout(c_features, p=self.dropout)

        # let method know its class features
        mc_own = torch.mm(mc_own_adj, c_features)                                                        

        # let method know feature of classes it calls
        mc_features = torch.cat((m_features, c_features), dim=0)
        mc_edges = mc_call_adj.indices().clone()
        mc_edges[1,:] += m_features.shape[0]
        mc_call = self.conv1(mc_features, mc_edges)
        mc_call = mc_call[:m_features.shape[0],:]

        # representation of method
        x_m = mc_own + mc_call

        # let class know its methods features
        cm_features = torch.cat((c_features, m_features), dim=0)
        cm_edges = mc_own_adj.indices().flip(0)
        cm_edges[1,:] += c_features.shape[0]
        cm_own = self.conv2(cm_features, cm_edges)
        cm_own = cm_own[:c_features.shape[0],:]

        # let class know features of methods who calls it
        cm_edges = mc_call_adj.indices().flip(0)
        cm_edges[1,:] += c_features.shape[0]
        cm_call = self.conv3(cm_features, cm_edges)
        cm_call = cm_call[:c_features.shape[0],:]

        # representation of class
        x_c = cm_own + cm_call

        # build ownship adjacency between method and class
        adj = torch.mm(x_m, x_c.t())                      

        # goes through softmax, as a method only belongs to a class
        adj = torch.softmax(adj, dim=1)

        return adj

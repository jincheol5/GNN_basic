import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


### GCN for Node Classification
class GCN_N(torch.nn.Module):
    def __init__(self,input_feature,output_feature):
        super().__init__()
        self.conv1 = GCNConv(input_feature, 16)
        self.conv2 = GCNConv(16, output_feature)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


### GNN for Link Prediction
class GCN_Encoder(torch.nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.conv1 = GCNConv(input_feature, 16)
        self.conv2 = GCNConv(16, input_feature)

    def forward(self, x,pos_edge_index):

        x = self.conv1(x, pos_edge_index)
        x = F.relu(x)
        x = self.conv2(x, pos_edge_index)

        return x

class Decoder(torch.nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.linear1=torch.nn.Linear(2*input_feature,input_feature)
        self.linear2=torch.nn.Linear(input_feature,1)

    def forward(self, x,pos_edge_index,neg_edge_index):
        
        forward_edge_index=torch.cat([pos_edge_index, neg_edge_index], dim=-1) # (2,num_pos_edges) + (2,num_neg_edges) = (2,num_pos_edges+num_neg_edges)

        src = forward_edge_index[0] # src=(num_pos_edges+num_neg_edges,)
        tar = forward_edge_index[1] # tar=(num_pos_edges+num_neg_edges,)

        z=torch.cat([x[src],x[tar]],dim=-1) # x[src], x[tar]=(num_pos_edges+num_neg_edges,input_features), z=(num_pos_edges+num_neg_edges,2*input_features)

        z=self.linear1(z)
        z = F.relu(z)
        z=self.linear2(z)

        return z # (num_pos_edges+num_neg_edges,1)

class GNN_L(torch.nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.encoder=GCN_Encoder(input_feature)
        self.decoder=Decoder(input_feature)

    def forward(self,x,pos_edge_index,neg_edge_index):

        x=self.encoder(x,pos_edge_index)
        z=self.decoder(x,pos_edge_index,neg_edge_index)

        return F.sigmoid(z)
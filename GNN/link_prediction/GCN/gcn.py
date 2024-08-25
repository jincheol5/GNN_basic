import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gcn1=GCNConv(input_dim,hidden_dim)
        self.gcn2=GCNConv(hidden_dim,output_dim)
        self.activation=torch.nn.ReLU()

    # 순전파
    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x=self.gcn1(x,edge_index)
        x=self.activation(x)
        x=self.gcn2(x,edge_index)
        return F.log_softmax(x, dim=1) # dim=1 -> 각 row마다 softmax 적용
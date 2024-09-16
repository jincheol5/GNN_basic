import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

### GCN Node Classification
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
    

### Reachability GNN
class Encoder(torch.nn.Module):
    def __init__(self,input_feature,output_feature):
        super().__init__()
        self.conv1 = GCNConv(input_feature, 16)
        self.conv2 = GCNConv(16, output_feature)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x

class Decoder(torch.nn.Module):
    def __init__(self,input_feature,output_feature):
        super().__init__()
        self.conv1 = GCNConv(input_feature, 16)
        self.conv2 = GCNConv(16, output_feature)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x

class R_GNN(torch.nn.Module):
    def __init__(self,input_feature,output_feature):
        super().__init__()
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        
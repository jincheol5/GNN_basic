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
    def __init__(self,input_feature):
        super().__init__()
        self.conv1 = GCNConv(input_feature, 16)
        self.conv2 = GCNConv(16, input_feature)

    def forward(self, x,edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x

class Decoder(torch.nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.linear1 = torch.nn.Linear(2*input_feature, input_feature)
        self.linear2 = torch.nn.Linear(input_feature, 1)

    def forward(self, x,R_index):
        src,tar=R_index

        z=torch.cat([x[src],x[tar]],dim=-1) # (num_labels,z_features)

        z=self.linear1(z)
        z = F.relu(z)
        z=self.linear2(z)

        return z # (num_labels,1)

class R_GNN(torch.nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.encoder=Encoder(input_feature)
        self.decoder=Decoder(input_feature)

    def forward(self, data,R_index):
        x, edge_index = data.x, data.edge_index

        x=self.encoder(x,edge_index) # output: x=(num_nodes,features)
        z=self.decoder(x,R_index) # output: z=(num_labels,1)

        return F.sigmoid(z)


import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

x=torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]],dtype=torch.float)

edge_index=torch.tensor([[0,1,2,3],[1,0,3,2]],dtype=torch.long)

data=Data(x=x,edge_index=edge_index)

data=train_test_split_edges(data=data)

print(data.train_pos_edge_index)

# print(data.train_neg_adj_mask)

# print(data.test_pos_edge_index)
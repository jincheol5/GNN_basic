import torch
import numpy as np

x_features=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]

x=torch.tensor(x_features)

data=[[0,2],[1,3]]

edge_label_index=torch.tensor(data)

src,tar=edge_label_index


z=torch.cat([x[src],x[tar]],dim=-1)

print(z.view(-1))

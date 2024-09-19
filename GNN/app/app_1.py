import torch
from GNN.data_load import DataLoader
from GNN.model import R_GNN
from GNN.model_train import Trainer

data_loader=DataLoader("Cora")

data=data_loader.load_R()

model=R_GNN(input_feature=data.num_node_features)

trainer=Trainer(model=model)

trainer.train_R(data=data,lr=0.01,epochs=100)

trainer.evaluate_R(data=data)
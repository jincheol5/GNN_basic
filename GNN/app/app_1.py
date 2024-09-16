import torch
from data_load import DataLoader
from model import GCN_N
from GNN.model_train import Trainer

data_loader=DataLoader("Cora")

data=data_loader.load_for_node_classification()

model=GCN_N(input_feature=data.num_node_features,output_feature=torch.unique(data.y).size(0))

trainer=Trainer(model=model)

trainer.train(data=data,lr=0.01,epochs=100)

trainer.evaluate(data=data)
import torch
from data_load import DataLoader
from model import GCN
from train import Trainer

data_loader=DataLoader("Cora")

data=data_loader.load_for_node_classification()

model=GCN(input_feature=data.num_node_features,output_feature=data.num_classes)

trainer=Trainer(model=model)

trainer.train(data=data,lr=0.01,epochs=100)

trainer.evaluate(data=data)
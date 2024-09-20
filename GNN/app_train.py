import torch
from data_load import DataLoad
from model import R_GNN
from model_train import Trainer

data_loader=DataLoad("Cora")

data,train_dataset,test_dataset=data_loader.load_R()

model=R_GNN(input_feature=data.num_node_features)

trainer=Trainer(model=model)

trainer.train_R(data=data,train_dataset=train_dataset)

trainer.evaluate_R(data=data,test_dataset=test_dataset)

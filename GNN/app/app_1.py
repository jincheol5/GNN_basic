import sys
import os
# 현재 파일의 상위 디렉토리(GNN) 경로를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from data_load import DataLoader
from model import R_GNN
from model_train import Trainer

data_loader=DataLoader("Cora")

data=data_loader.load_R()

model=R_GNN(input_feature=data.num_node_features)

trainer=Trainer(model=model)

trainer.train_R(data=data,lr=0.01,epochs=100)

trainer.evaluate_R(data=data)
import os
import random
import numpy as np
import torch
from utils.data_load import DataLoad
from model.model import GNN_L
from model.model_train import Trainer

### seed setting
seed= 42
random.seed(seed) # Python의 기본 random seed 설정
np.random.seed(seed) # NumPy의 random seed 설정
torch.manual_seed(seed) # PyTorch의 random seed 설정
os.environ["PYTHONHASHSEED"] = str(seed)
# CUDA 사용 시, 모든 GPU에서 동일한 seed 사용
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# PyTorch의 난수 생성 결정론적 동작 보장 (동일 연산 결과 보장)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False


data_loader=DataLoad("Cora")
data,train_neg_edge_index=data_loader.load_L()
model=GNN_L(input_feature=data.num_node_features)
model_trainer=Trainer(model=model)
model_trainer.train_L(x=data.x,pos_edge_index=data.train_pos_edge_index,neg_edge_index=train_neg_edge_index)
model_trainer.evaluate_L(x=data.x,pos_edge_index=data.test_pos_edge_index,neg_edge_index=data.test_neg_edge_index)
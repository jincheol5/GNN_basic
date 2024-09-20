import random
import numpy as np
import torch
from data_load import DataLoad
from model import R_GNN
from model_train import Trainer

seed= 42

random.seed(seed) # Python의 기본 random seed 설정
np.random.seed(seed) # NumPy의 random seed 설정
torch.manual_seed(seed) # PyTorch의 random seed 설정

# CUDA 사용 시, 모든 GPU에서 동일한 seed 사용
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# PyTorch의 난수 생성 결정론적 동작 보장 (동일 연산 결과 보장)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

data_loader=DataLoad("Cora")

data,train_dataset,test_dataset=data_loader.load_R()

model=R_GNN(input_feature=data.num_node_features)

trainer=Trainer(model=model)

trainer.train_R(data=data,train_dataset=train_dataset,epochs=10)

trainer.evaluate_R(data=data,test_dataset=test_dataset)

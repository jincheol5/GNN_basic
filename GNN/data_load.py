import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

class DataLoader:

    def __init__(self,dataset_name=None):
        self.dataset_name=dataset_name
        self.dataset_path=os.path.join("..", "data",self.dataset_name)

    def set_dataset_name(self,dataset_name):
        self.dataset_name=dataset_name
        self.dataset_path=os.path.join("..", "data", self.dataset_name)

    def load(self):
        self.x_df=pd.read_csv(os.path.join(self.dataset_path,"x.csv"))
        self.edge_index_df=pd.read_csv(os.path.join(self.dataset_path,"edge_index.csv"))
        self.y_train_df=pd.read_csv(os.path.join(self.dataset_path,"y_train.csv"))
        self.y_val_df=pd.read_csv(os.path.join(self.dataset_path,"y_val.csv"))
        self.y_test_df=pd.read_csv(os.path.join(self.dataset_path,"y_test.csv"))

        ### edge_index
        source_edge_index=self.edge_index_df['source'].values # .values = numpy.ndarray 반환
        target_edge_index=self.edge_index_df['target'].values

        edge_index_ndarray = np.array([source_edge_index, target_edge_index]) # 빠른 연산을 위해 ndarray 두개를 하나의 ndarray로 변환 후 tensor로 변환
        
        edge_index = torch.tensor(edge_index_ndarray, dtype=torch.long)

        ### node features
        x_features_pandas_series=self.x_df.apply(lambda x: x.to_numpy(), axis=1) # axis=1 -> 행 단위로 작업 수행
        x_features_ndarray=np.vstack(x_features_pandas_series) # .vstack() -> 여러 개의 ndarray를 하나의 2차원 배열로 쌓을 수 있다
        x_features=torch.tensor(x_features_ndarray,dtype=torch.float)

        ### set Data
        self.graph=Data(x=x_features,edge_index=edge_index)

        return self.graph


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

    def load_for_link_prediction(self):
        self.x_df=pd.read_csv(os.path.join(self.dataset_path,"x.csv"))
        self.edge_index_df=pd.read_csv(os.path.join(self.dataset_path,"edge_index.csv"))
        
        ### node features
        x_features_pandas_series=self.x_df.apply(lambda x: x.to_numpy(), axis=1) # axis=1 -> 행 단위로 작업 수행
        x_features_ndarray=np.vstack(x_features_pandas_series) # .vstack() -> 여러 개의 ndarray를 하나의 2차원 배열로 쌓을 수 있다
        x_features=torch.tensor(x_features_ndarray,dtype=torch.float)


        ### edge_index
        source_edge_index=self.edge_index_df['source'].values # .values = numpy.ndarray 반환
        target_edge_index=self.edge_index_df['target'].values

        edge_index_ndarray = np.array([source_edge_index, target_edge_index]) # 빠른 연산을 위해 ndarray 두개를 하나의 ndarray로 변환 후 tensor로 변환
        
        edge_index = torch.tensor(edge_index_ndarray, dtype=torch.long)

        
        ### set Data
        self.graph=Data(x=x_features,edge_index=edge_index)

        return self.graph

    def load_for_node_classification(self):
        self.x_df=pd.read_csv(os.path.join(self.dataset_path,"x.csv"))
        self.edge_index_df=pd.read_csv(os.path.join(self.dataset_path,"edge_index.csv"))
        self.y_train_df=pd.read_csv(os.path.join(self.dataset_path,"y_train.csv"))
        self.y_val_df=pd.read_csv(os.path.join(self.dataset_path,"y_val.csv"))
        self.y_test_df=pd.read_csv(os.path.join(self.dataset_path,"y_test.csv"))
        
        ### node features
        x_features_pandas_series=self.x_df.apply(lambda x: x.to_numpy(), axis=1) # axis=1 -> 행 단위로 작업 수행
        x_features_ndarray=np.vstack(x_features_pandas_series) # .vstack() -> 여러 개의 ndarray를 하나의 2차원 배열로 쌓을 수 있다
        x_features=torch.tensor(x_features_ndarray,dtype=torch.float)


        ### edge_index
        source_edge_index=self.edge_index_df['source'].values # .values = numpy.ndarray 반환
        target_edge_index=self.edge_index_df['target'].values

        edge_index_ndarray = np.array([source_edge_index, target_edge_index]) # 빠른 연산을 위해 ndarray 두개를 하나의 ndarray로 변환 후 tensor로 변환
        
        edge_index = torch.tensor(edge_index_ndarray, dtype=torch.long)
        

        ### node label 
        y_train=self.y_train_df['label'].values # .values = numpy.ndarray 반환
        y_val=self.y_val_df['label'].values
        y_test=self.y_test_df['label'].values

        y_label = np.concatenate([y_train, y_val, y_test])

        ### mask
        num_nodes=self.x_df.shape[0] # row 개수 = node 수

        train_mask=torch.zeros(num_nodes, dtype=torch.bool) # false로 초기화
        val_mask=torch.zeros(num_nodes, dtype=torch.bool)
        test_mask=torch.zeros(num_nodes, dtype=torch.bool)

        for index in self.y_train_df['index'].values:
            train_mask[index]=True
        
        for index in self.y_val_df['index'].values:
            val_mask[index]=True
        
        for index in self.y_test_df['index'].values:
            test_mask[index]=True

        ### set Data
        self.graph=Data(x=x_features,edge_index=edge_index,y=y_label,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)

        return self.graph


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

    def load_L(self):
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

    def load_N(self):
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
        num_nodes=self.x_df.shape[0] # row 개수 = node 수

        y_label=tensor = torch.full((num_nodes,), -1) # 값 -1로 채워진 num_nodes 크기의 1차원 tensor
        # 준지도 학습 노드 분류의 경우, 값이 주어지지 않은 노드에 대한 라벨은 -1로 지정한다
        # mask 된 노드들에 대해서만 학습을 하기 때문에, 라벨이 없는 노드의 y 값은 손실 계산 시 사용되지 않으므로, -1은 학습에 영향을 미치지 않는다
        # y_train_df['index'].values로 y_label tensor의 특정 위치에 접근 후, 그 위치에 y_train_df['label']에 저장된 값을 할당
        y_label[self.y_train_df['index'].values] = torch.tensor(self.y_train_df['label'].values, dtype=torch.long)
        y_label[self.y_val_df['index'].values] = torch.tensor(self.y_val_df['label'].values, dtype=torch.long)
        y_label[self.y_test_df['index'].values] = torch.tensor(self.y_test_df['label'].values, dtype=torch.long)

        ### mask
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

    def load_R(self):
        self.x_df=pd.read_csv(os.path.join(self.dataset_path,"x.csv"))
        self.edge_index_df=pd.read_csv(os.path.join(self.dataset_path,"edge_index.csv"))
        self.train_reachability_df=pd.read_csv(os.path.join(self.dataset_path,"train_reachability.csv")) # source target label
        self.test_reachability_df=pd.read_csv(os.path.join(self.dataset_path,"test_reachability.csv"))

        ### node features
        x_features_pandas_series=self.x_df.apply(lambda x: x.to_numpy(), axis=1) # axis=1 -> 행 단위로 작업 수행
        x_features_ndarray=np.vstack(x_features_pandas_series) # .vstack() -> 여러 개의 ndarray를 하나의 2차원 배열로 쌓을 수 있다
        x_features=torch.tensor(x_features_ndarray,dtype=torch.float)

        ### edge_index
        source_edge_index=self.edge_index_df['source'].values # .values = numpy.ndarray 반환
        target_edge_index=self.edge_index_df['target'].values

        edge_index_ndarray = np.array([source_edge_index, target_edge_index]) # 빠른 연산을 위해 ndarray 두개를 하나의 ndarray로 변환 후 tensor로 변환
        
        edge_index = torch.tensor(edge_index_ndarray, dtype=torch.long)

        ### train_reachability
        source_train_reachability=self.train_reachability_df['source'].values # .values = numpy.ndarray 반환
        target_train_reachability=self.train_reachability_df['target'].values
        label_train_reachability=self.train_reachability_df['label'].values

        train_reachability_ndarray=np.array([source_train_reachability,target_train_reachability])
        train_reachability_label_ndarray=np.array(label_train_reachability)

        train_R=torch.tensor(train_reachability_ndarray, dtype=torch.long)
        train_R_label=torch.tensor(train_reachability_label_ndarray, dtype=torch.long)

        ### test_reachability
        source_test_reachability=self.test_reachability_df['source'].values # .values = numpy.ndarray 반환
        target_test_reachability=self.test_reachability_df['target'].values
        label_test_reachability=self.test_reachability_df['label'].values

        test_reachability_ndarray=np.array([source_test_reachability,target_test_reachability])
        test_reachability_label_ndarray=np.array(label_test_reachability)

        test_R=torch.tensor(test_reachability_ndarray, dtype=torch.long)
        test_R_label=torch.tensor(test_reachability_label_ndarray, dtype=torch.long)

        ### set Data
        self.graph=Data(x=x_features,edge_index=edge_index,train_R=train_R,train_R_label=train_R_label,test_R=test_R,test_R_label=test_R_label)

        return self.graph
import networkx as nx
import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self):
        self.graph=nx.DiGraph()
    
    def load_static_graph(self,dataset_name=None):
        self.dataset_name=dataset_name
        self.dataset_path=os.path.join("..", "data", self.dataset_name)

        self.x_df=pd.read_csv(os.path.join(self.dataset_path,"x.csv"))
        self.edge_index_df=pd.read_csv(os.path.join(self.dataset_path,"edge_index.csv"))

        # 0부터 n까지의 node를 input
        self.graph.add_nodes_from(range(self.x_df.shape[0])) 
        
        # edge input
        for row in self.edge_index_df.itertuples():
            self.graph.add_edge(row.source,row.target)

    def load_reachability(self):
        nodes=list(self.graph.nodes())
        train_size = int(len(nodes) * 0.1) # 노드의 10% 학습용으로 사용 
        train_nodes=np.random.choice(nodes, train_size, replace=False) # 노드의 10% 랜덤하게 선택

        results=[]

        # compute reachability of all pairs nodes
        for source in nodes:
            for target in nodes:
                label=0
                if nx.has_path(self.graph,source,target): label=1
                results.append([source,target,label])

        result_df = pd.DataFrame(results, columns=['source', 'target', 'label'])
        train_data_df=result_df[result_df['source'].isin(train_nodes)]
        test_data_df=result_df[~result_df['source'].isin(train_nodes)]

        train_data_df.to_csv(os.path.join(self.dataset_path,"train_reachability.csv"), index=False)
        test_data_df.to_csv(os.path.join(self.dataset_path,"test_reachability.csv"), index=False)
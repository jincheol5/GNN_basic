import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self,model=None):
        self.model=model
    
    def set_model(self,model):
        self.model=model

    def train_N(self,data,lr=0.01,epochs=100):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        data.to(device)
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in tqdm(range(epochs),desc="Training..."):
            optimizer.zero_grad()
            output=self.model(data)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    def evaluate_N(self,data):
        self.model.eval()
        pred = self.model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        
        print(f"acc: {acc * 100:.2f}%")

    def train_R(self,data,train_dataset,lr=0.01,epochs=100,batch_size=64):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        data.to(device)
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()

        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        for epoch in tqdm(range(epochs),desc="Training..."):
            for reachability_edge_index,reachability_edge_label in train_dataloader:

                reachability_edge_index.to(device) # (batchsize,2)
                reachability_edge_label.to(device) # (batchsize)
                optimizer.zero_grad()
                output=self.model(data,reachability_edge_index) # output: z=(batchsize,1)
                loss=F.binary_cross_entropy(output,reachability_edge_label) # reachability_edge_label=(batchsize,1)
                loss.backward()
                optimizer.step()
    
    def evaluate_R(self,data,test_dataset,batch_size=64):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        data.to(device)

        test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

        correct_sum=0
        num_labels=0
        with torch.no_grad():
            self.model.eval()
            for reachability_edge_index,reachability_edge_label in test_dataloader:
                reachability_edge_index.to(device)
                reachability_edge_label.to(device)
                pred = (self.model(data,reachability_edge_index)>=0.5).long() # 0.5보다 크면 1, 작으면 0, output: pred=(num_labels,1)
                correct = (pred == reachability_edge_label).sum()
                correct_sum+=correct
                num_labels+=reachability_edge_label.sum()
        
        acc = int(correct_sum) / int(num_labels)

        print(f"acc: {acc * 100:.2f}%")
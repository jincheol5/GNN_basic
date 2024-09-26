import torch
import torch.nn.functional as F
import torchmetrics
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

    def train_L(self,x,pos_edge_index,neg_edge_index,lr=0.01,epochs=100):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        x=x.to(device)
        pos_edge_index=pos_edge_index.to(device)
        neg_edge_index=neg_edge_index.to(device)

        total_edge_size=pos_edge_index.size(1) + neg_edge_index.size(1)
        edge_labels=torch.zeros(total_edge_size, dtype=torch.float, device=device)
        edge_labels[:pos_edge_index.size(1)] = 1
        edge_labels=edge_labels.unsqueeze(1) # (total_edge_num,) -> (total_edge_num,1)

        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in tqdm(range(epochs),desc="Training..."):
            optimizer.zero_grad()
            output=self.model(x,pos_edge_index,neg_edge_index) # output=(total_edge_num,1)
            loss=F.binary_cross_entropy(output,edge_labels) # output=(total_edge_num,1), edge_labels=(total_edge_num,1)
            loss.backward()
            optimizer.step()

    def evaluate_L(self,x,pos_edge_index,neg_edge_index):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        x=x.to(device)
        pos_edge_index=pos_edge_index.to(device)
        neg_edge_index=neg_edge_index.to(device)

        total_edge_size=pos_edge_index.size(1) + neg_edge_index.size(1)
        edge_labels=torch.zeros(total_edge_size, dtype=torch.float, device=device)
        edge_labels[:pos_edge_index.size(1)] = 1
        edge_labels=edge_labels.unsqueeze(1) # (total_edge_num,) -> (total_edge_num,1)


        correct_sum=0

        # torchmetrics로 AUROC 계산을 초기화
        auroc = torchmetrics.AUROC(task='binary').to(device)

        with torch.no_grad():
            self.model.eval()
            output=self.model(x,pos_edge_index,neg_edge_index) # output=(total_edge_num,1)
            pred=(output>=0.5).long() # 0.5보다 크면 1, 작으면 0, output: pred=(total_edge_num,1)
            correct = (pred == edge_labels).sum().item()
            correct_sum+=correct

            # AUC 계산을 위해 1D (batchsize,)로 변환 후 업데이트 -> 대규모 데이터에 대한 점진적 업데이트
            auroc.update(output.squeeze(), edge_labels.squeeze())

        # Model Accuracy
        acc=correct_sum/total_edge_size
        print(f"acc: {acc * 100:.2f}%")

        # Model AUC
        auc = auroc.compute()
        print(f"AUC: {auc:.4f}")
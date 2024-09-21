import torch
import torch.nn.functional as F
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

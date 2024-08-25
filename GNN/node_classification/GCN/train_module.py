import torch
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self,model):
        self.model=model
        self.model.train()
        self.optimizer=optim.Adam(self.model.parameters(), lr=0.001) # 최적화 알고리즘, model의 parameter들을 참조 
        self.criterion=torch.nn.CrossEntropyLoss() # 손실 함수 
    
    def training(self,data,epochs):
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad() # 이전 단계에서 계산된 기울기(gradient) 초기화
            pred=self.model(data) # 순전파 forward 실행, pytorch의 자동 미분 엔진 Autograd가 모델의 순전파 연산 동안 연산 그래프를 자동으로 생성, 이후 이 그래프를 통해 역전파 수행
            loss=self.criterion(pred[data.train_mask],data.y[data.train_mask])
            loss.backward() # 오류 역전파 -> 각 파라미터에 대한 기울기 계산
            self.optimizer.step() # 계산된 기울기(gradient) 기반으로 model parameter 업데이트 

            if epoch == epochs:
                print(f'Loss: {loss.item()}')

    
import torch

class Trainer:
    def __init__(self,model=None):
        self.model=model
    
    def set_model(self,model):
        self.model=model
import torch
import torch.nn as nn

def train_model(model, trainloader):
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
    
    # Put the Model in training mode
    model.train()
    # Training Code
    
    
def test_model(model):
    
    # Put the Model in Testing Mode
    model.eval()
    
    # Testing Code
import torch
import torch.nn as nn
from src.dataset import train_loader, test_loader
LOCAL_EPOCHS = 2
def train_model(model, trainloader):
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Put the Model in training mode
    model.train()
    # Training Code
    for epoch in range(LOCAL_EPOCHS):
        for images, labels in train_loader:
            
            # Move to GPU
            
            # images = images.to(device)
            # labels = labels.to(device)
            
            # Reset gradients to zero for each batch
            optimizer.zero_grad()
            
            # Forward Inference on the input images to get model predictions
            outputs = model(images)
            
            # Loss using CrossEntropyLoss 
            loss = loss_fn(outputs, labels)
            
            # Compute the gradients of the loss (Rn only one layer as all other layers are frozen)
            loss.backward()
            
            # Propagate the gradients back and modify the weights
            optimizer.step()           
    
def test_model(model):
    
    # Put the Model in Testing Mode
    model.eval()
    
    # Testing Code
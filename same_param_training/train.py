import torch
import torch.nn as nn

def train_model(model, learning_rate, weight_decay , local_epochs, device, train_loader):
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Put the Model in training mode
    model.train()
    running_loss = 0.0
    # Training Code
    for epoch in range(local_epochs):
        for images, labels in train_loader:
            
            # Move to GPU
            images = images.to(device)
            labels = labels.to(device)
            
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
            running_loss += loss.item()
    avg_train_loss = running_loss / (local_epochs * len(train_loader))       
    return avg_train_loss

def test_model(model, device, val_loader):
    
    # Put the Model in Validation Mode
    loss_fn = nn.CrossEntropyLoss()
    correct , loss = 0 , 0.0
    model.eval()
    
    with torch.no_grad():
        for images, labels in val_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss += loss_fn(outputs, labels).item()
            correct += (torch.max(outputs.data,1)[1] == labels).sum().item()
            
    accuracy = correct / len(val_loader.dataset)
    loss = loss / len(val_loader)
    return loss, accuracy

import torch
import torch.nn as nn

def train_model(model, learning_rate, weight_decay, local_epochs, device, train_loader, proximal_mu: float = 0.0):

    # Only regularize/optimize params that are actually trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Snapshot global weights before local training starts
    global_params = [p.detach().clone() for p in trainable_params]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    model.train()
    running_loss = 0.0
    for epoch in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            task_loss = loss_fn(outputs, labels)
            loss = task_loss

            if proximal_mu > 0:
                proximal_term = sum(
                    (local_p - global_p).norm(2) ** 2
                    for local_p, global_p in zip(trainable_params, global_params)
                )
                loss = loss + (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            running_loss += task_loss.item()

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
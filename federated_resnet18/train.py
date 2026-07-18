import torch 
import torch.nn as nn

def train_model(model, learning_rate, weight_decay, local_epochs, device, train_loader):
    loss_fn= nn.CrossEntropyLoss()  #caluclates the mean of all the losses
    optimizer=torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), #defines which weights to actually update
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model.train()
    running_loss=0.0

    for epoch in range(local_epochs):
        for images, labels in train_loader:
            images= images.to(device)
            labels=labels.to(device)

            optimizer.zero_grad() #clears gradients
            outputs= model(images)
            loss= loss_fn(outputs,labels)  #loss_fn gets 32 predictions and 32 targets at once and calculates the individual losses and stores them

            loss.backward()
            optimizer.step()

            running_loss+= loss.item()

    avg_train_loss= running_loss/ (local_epochs* len(train_loader))
    return avg_train_loss


def test_model(model, device, val_loader):
    loss_fn= nn.CrossEntropyLoss()
    correct, loss= 0, 0.0
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images= images.to(device)
            labels=labels.to(device)

            outputs= model(images)
            loss+= loss_fn(outputs, labels).item()
            correct+=(torch.max(outputs.data, 1)[1]==labels).sum().item()

    accuracy= correct/len(val_loader.dataset)
    loss=loss/len(val_loader)
    return loss,accuracy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

PROCESSED = Path(r"C:\Users\Aspire\Desktop\Minor Project\Comparative-Study-of-Federated-Learning-and-Centralised-Learning\dataset\preprocessed")
# Get transforms to apply on the image
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
# Define the train data loader
def get_train_loader(batch_size = 32):
    dataset = datasets.ImageFolder(
        PROCESSED / "Train",
        transform=get_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
# Define the test data loader
def get_test_loader(batch_size = 32):
    dataset = datasets.ImageFolder(
        PROCESSED / "Test",
        transform=get_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
# Define the validation data loader
def get_val_loader(batch_size = 32):
    dataset = datasets.ImageFolder(
        PROCESSED / "Val",
        transform=get_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
def get_client_loader(client_id, n_clients, mode, batch_size=32):
    dataset = datasets.ImageFolder(
        PROCESSED / f"clients_{n_clients}_{mode}" / f"client_{client_id}",
        transform=get_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
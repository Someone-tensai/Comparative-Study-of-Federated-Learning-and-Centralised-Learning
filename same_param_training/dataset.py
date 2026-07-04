from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.model_selection import train_test_split

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
    

def get_client_loader(client_id, n_clients, mode, batch_size=32, val_fraction=0.1):
    
    dataset = datasets.ImageFolder(
        PROCESSED / f"clients_{n_clients}_{mode}" / f"client_{client_id}",
        transform=get_transform(),
    )

    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=dataset.targets,
        random_state=42,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

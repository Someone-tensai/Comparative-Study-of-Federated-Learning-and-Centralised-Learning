import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# set ur own working directory root (Temporary Fix)
PROCESSED = Path(r"C:\Users\Aspire\Desktop\Minor Project\Comparative-Study-of-Federated-Learning-and-Centralised-Learning\dataset\preprocessed")
# Get transforms to apply on the image
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_plain_loaders(batch_size=32):
    train_dir = '../dataset/Epic and CSCR hospital Dataset/Train'
    test_dir = '../dataset/Epic and CSCR hospital Dataset/Test'

    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_transforms())
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_transforms())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
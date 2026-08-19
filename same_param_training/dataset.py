from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from common.config import PROCESSED

# FIX: previously there was a single get_transform() used for train, test,
# AND the validation split inside get_client_loader(). It included
# RandomRotation(15°), which meant test/val accuracy was being measured on
# randomly-rotated images — noisy, biased, and non-reproducible between
# runs even with a fixed seed on the split itself.
#
# Now there are two transforms: one for training (augmented) and one for
# evaluation (deterministic — no augmentation).

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Kept for backwards compatibility with any code importing get_transform()
# directly (e.g. visualize_gradcam.py uses it to build a single-image input
# tensor for inference — that should also be non-augmented).
def get_transform():
    return get_eval_transform()


# Define the train data loader
def get_train_loader(batch_size=32):
    dataset = datasets.ImageFolder(
        PROCESSED / "Train",
        transform=get_train_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the test data loader
def get_test_loader(batch_size=32):
    dataset = datasets.ImageFolder(
        PROCESSED / "Test",
        transform=get_eval_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_client_loader(client_id, n_clients, mode, augmented, batch_size=32, val_fraction=0.1):

    folder = f"clients_{n_clients}_augmented_{mode}" if augmented else f"clients_{n_clients}_{mode}"
    path = Path(folder) / f"client_{client_id}"

    # FIX: train and val subsets used to come from ONE ImageFolder built with
    # the augmented transform, so val images were being rotated too. We now
    # build two ImageFolder instances over the same path — one with the
    # train transform, one with the eval transform — and Subset each with
    # the matching indices. Same underlying files, different transform per
    # split.
    train_base = datasets.ImageFolder(PROCESSED / path, transform=get_train_transform())
    eval_base = datasets.ImageFolder(PROCESSED / path, transform=get_eval_transform())

    indices = list(range(len(train_base)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=train_base.targets,
        random_state=42,
    )

    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(eval_base, val_idx)

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
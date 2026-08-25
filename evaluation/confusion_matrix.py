# evaluate_models.py

import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from same_param_training.models import our_model
from same_param_training.dataset import get_test_loader


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

MODEL_DIR = REPO_ROOT / "models"
OUTPUT_DIR = BASE_DIR

CM_DIR = OUTPUT_DIR / "confusion_matrices"
DATA_DIR = OUTPUT_DIR / "confusion_data"

BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SETUP
# ============================================================

CM_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEST DATA
# ============================================================

test_loader = get_test_loader(batch_size=BATCH_SIZE)

class_names = test_loader.dataset.classes
num_classes = len(class_names)

print(f"Device: {DEVICE}")
print(f"Classes: {class_names}")
print(f"Test samples: {len(test_loader.dataset)}")
print()


# ============================================================
# MODEL NAME DETECTION
# ============================================================

def get_model_name(filename):
    """
    Extract the architecture name from your run filename.

    Example:
        resnet18-layer4_out-...pth
            -> resnet18

        resnet50-layer4_out-...pth
            -> resnet50
    """

    name = filename.lower()

    if "resnet18" in name:
        return "resnet18"

    if "resnet50" in name:
        return "resnet50"

    if "vgg16" in name:
        return "vgg16"

    raise ValueError(
        f"Could not determine model architecture from filename: {filename}"
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model):

    model.eval()

    all_predictions = []
    all_targets = []

    for images, targets in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_targets.extend(targets.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    return all_targets, all_predictions


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def save_confusion_matrix(cm, class_names, output_path, model_name):

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=f"Confusion Matrix — {model_name}",
    )

    # Add values inside cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    # SVG preserves the text/vector graphics nicely
    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(targets, predictions):

    accuracy = accuracy_score(targets, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(num_classes),
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),

        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),

        "weighted_precision": float(
            np.average(precision, weights=support)
        ),
        "weighted_recall": float(
            np.average(recall, weights=support)
        ),
        "weighted_f1": float(
            np.average(f1, weights=support)
        ),

        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
    }


# ============================================================
# PROCESS ALL MODELS
# ============================================================

model_files = sorted(MODEL_DIR.glob("*.pth"))

if not model_files:
    raise RuntimeError(
        f"No .pth files found in {MODEL_DIR.resolve()}"
    )


summary = []

print(f"Found {len(model_files)} model(s)")
print("=" * 80)


for model_path in model_files:

    print(f"\nEvaluating: {model_path.name}")

    # --------------------------------------------------------
    # Determine architecture
    # --------------------------------------------------------

    model_name = get_model_name(model_path.name)

    print(f"Architecture: {model_name}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = our_model(model_name)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    targets, predictions = evaluate(model)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(num_classes),
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    output_name = model_path.stem

    cm_path = CM_DIR / f"{output_name}.svg"

    save_confusion_matrix(
        cm,
        class_names,
        cm_path,
        output_name,
    )

    # --------------------------------------------------------
    # Save raw data
    # --------------------------------------------------------

    result = {
        "model_file": model_path.name,
        "architecture": model_name,
        "classes": class_names,
        "confusion_matrix": cm.tolist(),
        "metrics": metrics,
    }

    json_path = DATA_DIR / f"{output_name}.json"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary.append({
        "model": output_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    })

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Saved:       {cm_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    f"{'Model':<55}"
    f"{'Accuracy':>10}"
    f"{'Macro F1':>10}"
)

print("-" * 80)

for result in sorted(
    summary,
    key=lambda x: x["accuracy"],
    reverse=True,
):

    print(
        f"{result['model']:<55}"
        f"{result['accuracy']:>10.4f}"
        f"{result['macro_f1']:>10.4f}"
    )

print("=" * 80)

print(f"\nConfusion matrices: {CM_DIR.resolve()}")
print(f"Raw metrics:        {DATA_DIR.resolve()}")
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from models import our_model
from utils.gradcam import GradCam, get_target_layer
from same_param_training.dataset import get_transform, get_test_loader

from common.config import PROCESSED, CLASSES

MODEL_NAME = "resnet50"
WEIGHTS_PATH = "weights.pth"

model = our_model(MODEL_NAME, freeze_backbone=True)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

target_layer = get_target_layer(model, MODEL_NAME)
cam_extractor = GradCam(model, target_layer)


def load_image_by_path(img_path: Path):
    pil_img = Image.open(img_path).convert("RGB")  # duplicate gray -> 3ch
    input_tensor = get_transform()(pil_img).unsqueeze(0)  # [1, 3, H, W]
    return input_tensor


def load_image_from_test_loader(index: int = 0):
    """
    Pull a single image from the test set by index. Returns the
    transformed input tensor, its ground-truth label, its source file
    path (needed for the overlay), and the full class name list
    (so callers don't need to rebuild the loader just for that).
    """
    test_loader = get_test_loader(batch_size=1)
    dataset = test_loader.dataset  # ImageFolder instance

    img_path, label_idx = dataset.samples[index]  # (path, class_idx)
    class_names = dataset.classes
    true_label = class_names[label_idx]

    input_tensor, _ = dataset[index]          # apply the same transform
    input_tensor = input_tensor.unsqueeze(0)  # add batch dim -> [1, 3, H, W]

    return input_tensor, true_label, Path(img_path), class_names


def save_overlay(img_path: Path, heatmap: np.ndarray, out_path: str):
    original_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    original_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))
    original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_color, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(out_path, overlay)
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":

    # ---- Option A: pick a specific image by path ----
    # img_path = PROCESSED / "Test" / CLASSES[0] / "example.jpg"  # <-- edit filename
    # input_tensor = load_image_by_path(img_path)
    # heatmap, pred_class = cam_extractor.generate(input_tensor)
    # print(f"Predicted class index: {pred_class} ({CLASSES[pred_class]})")
    # save_overlay(img_path, heatmap, "gradcam_output.jpg")

    # ---- Option B: grab a test image by index, with ground-truth label ----
    input_tensor, true_label, img_path, class_names = load_image_from_test_loader(index=0)

    heatmap, pred_class = cam_extractor.generate(input_tensor)
    print(f"Predicted: {class_names[pred_class]}  |  Ground truth: {true_label}")

    save_overlay(img_path, heatmap, "gradcam_output.jpg")
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        
        # Runs everytime the target_layer finishes a forward pss
        def forward_hook(module, input, output):
            self.activations = output
        
        # Runs when gradients flow back through that layer
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate(self, input_tensor, class_idx = None):
        
        # Evaluation Mode for Deterministic Behaviour
        self.model.eval()
        
        # If frozen backbone, then we make sure gradients can flow
        # on inference to reach the target layer
         
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        
        # Specific Class to Explain 
        # By Default its the class the model predicted
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        
        # Triggers the Forward Hook (Inference)
        loss = output[0, class_idx]
        
        # Triggers the Backward Hook (Gradients)
        loss.backward()
        
        # Average of the gradient across one channel (Average of Pixels)
        weights = self.gradients.mean(dim=(2,3), keepdim = True)
        
        # Multiple by importance weight and sum the channels to get one spatial map
        cam = (weights * self.activations).sum(dim=1, keepdim = True)
        
        # Choose only features that increase the prediction accuracy(Drop negative influences)
        cam = F.relu(cam)
        
        # Change shape to make it 2D
        # Detach it from autograd graph 
        # Convert to numpy for CV
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Upsample the target layer feature map into image size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        #  Min-Max Normalisation to render as a heatmap (1e-8 to avoide /0 problems)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
        
def get_target_layer(model, name):
    if name == "resnet50" or name == "resnet18":
        return model.layer4[-1]          # last residual block
    if name == "vgg16":
        return model.features[28]        # last conv layer, before final ReLU/MaxPool
    raise ValueError(f"No target layer defined for {name}")
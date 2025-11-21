import numpy as np
from captum.attr import IntegratedGradients

# TODO: Add type hinting
def compute_integrated_gradients(model, images, labels, device='cpu', steps=200):
    """Compute IG attributions for a batch of images"""
    print(f"Computing IG on device: {device}")  # Add this line
    print(f"Number of images: {images.shape[0]}, Steps: {steps}")  # Add this too
    ig = IntegratedGradients(model)
    attr = ig.attribute(images, target=labels, n_steps=steps)
    return attr.squeeze().detach().cpu().numpy()
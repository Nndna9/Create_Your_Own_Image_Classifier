import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np

# Import utility functions from utils.py
from utils import process_image, load_checkpoint

def predict_image(image_path, checkpoint, top_k, category_names, gpu):
    # Load the checkpoint
    model = load_checkpoint(checkpoint)

    # Move model to GPU if available
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Process the image
    img_tensor = process_image(image_path)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
    probs = torch.exp(output)
    top_probs, top_classes = probs.topk(top_k)

    # Load category names
    if category_names:
        with open(category_names, '
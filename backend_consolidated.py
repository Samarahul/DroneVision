# backend_consolidated.py
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.backend as K
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ===== U-Net Functions =====
def preprocess_unet_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def load_unet_model(model_name="unet.keras"):
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    return load_model(model_path, custom_objects={
        'soft_dice_loss': soft_dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef
    })

def predict_segmentation(unet_model, image_tensor, threshold=0.1):
    prediction = unet_model.predict(image_tensor)
    return (prediction > threshold).astype(np.uint8)

def predict_with_unet(image_path, output_path):
    model = load_unet_model()
    input_tensor = preprocess_unet_image(image_path)
    prediction = predict_segmentation(model, input_tensor)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, np.squeeze(prediction), cmap='gray')
    return output_path

# ===== Comparison Functions =====
def compute_iou(img1, img2):
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    return intersection / union if union != 0 else 1.0

def compute_progress(img1, img2):
    area1 = np.count_nonzero(img1)
    area2 = np.count_nonzero(img2)
    return ((area2 - area1) / area1 * 100) if area1 != 0 else (100.0 if area2 > 0 else 0.0)

def preprocess_mask(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 127).astype(np.uint8)

def compare_segmented_images(img1, img2):
    img1 = preprocess_mask(img1)
    img2 = preprocess_mask(img2)
    return {
        "iou": compute_iou(img1, img2),
        "progress_percent": compute_progress(img1, img2)
    }

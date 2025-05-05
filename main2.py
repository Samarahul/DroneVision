import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import gradio as gr
from datetime import datetime
from backend_consolidated import predict_with_unet, compare_segmented_images

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

def is_image_sharp(image, threshold=20.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold, laplacian_var

def validate_metadata(coords1, coords2, alt1, alt2, ts1, ts2):
    try:
        lat1, lon1 = map(float, coords1.strip().split(","))
        lat2, lon2 = map(float, coords2.strip().split(","))
        alt1 = float(alt1)
        alt2 = float(alt2)
        ts1 = datetime.strptime(ts1, "%Y-%m-%d %H:%M:%S")
        ts2 = datetime.strptime(ts2, "%Y-%m-%d %H:%M:%S")

        # Coordinate difference
        if abs(lat1 - lat2) > 0.01 or abs(lon1 - lon2) > 0.01:
            return False, "Coordinates are too far apart."

        # Altitude difference
        if abs((alt1 - alt2) >10 or ((alt1<100 or alt1>300) and (alt2<100 or alt2>300))):
            return False, "Altitude error."

        # Timestamp order
        if ts1 >= ts2:
            return False, "Image 1 timestamp should be earlier than Image 2."

        return True, "Validation passed."

    except Exception as e:
        return False, f"Metadata validation failed: {str(e)}"

def process_and_compare(image1, image2, coords1, coords2, alt1, alt2, ts1, ts2):
    try:
        # Convert to OpenCV format
        img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

        # Sharpness check
        img1_sharp, sharpness1 = is_image_sharp(img1)
        img2_sharp, sharpness2 = is_image_sharp(img2)

        if not img1_sharp or not img2_sharp:
            return f"One or both images are blurry. Sharpness scores: Image 1 = {sharpness1:.2f}, Image 2 = {sharpness2:.2f}", None, None

        # Metadata validation
        valid, message = validate_metadata(coords1, coords2, alt1, alt2, ts1, ts2)
        if not valid:
            return message, None, None

        # Save temp images
        temp_path1 = os.path.join(output_folder, "temp_img1.jpg")
        temp_path2 = os.path.join(output_folder, "temp_img2.jpg")
        cv2.imwrite(temp_path1, img1)
        cv2.imwrite(temp_path2, img2)

        # Predict using U-Net
        segmented_outputs = []
        segmented_images_paths = []

        for idx, path in enumerate([temp_path1, temp_path2]):
            filename = os.path.basename(path)
            output_path = os.path.join(output_folder, f"unet_output_{filename}")
            pred_path = predict_with_unet(path, output_path)

            if isinstance(pred_path, str) and os.path.exists(pred_path):
                mask_array = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                segmented_outputs.append(np.squeeze(mask_array))
                segmented_images_paths.append(pred_path)
            else:
                return "Prediction failed for one or both images.", None, None

        if len(segmented_outputs) == 2:
            results = compare_segmented_images(segmented_outputs[0], segmented_outputs[1])
            iou = f"IoU: {results['iou']:.4f}"
            progress = f"Progress: {results['progress_percent']:+.2f}%"

            # Description of road development
            description = generate_road_development_description(segmented_outputs[0], segmented_outputs[1])
            return f"{iou}\n{progress}\n{description}", segmented_images_paths[0], segmented_images_paths[1]
        else:
            return "Need exactly 2 images to compare.", None, None

    except Exception as e:
        return f"Error: {str(e)}", None, None

def generate_road_development_description(img1, img2):
    new_roads = np.logical_and(img2 == 1, img1 == 0)
    removed_roads = np.logical_and(img1 == 1, img2 == 0)

    new_roads_area = np.sum(new_roads)
    removed_roads_area = np.sum(removed_roads)

    if new_roads_area > 0:
        description = f"Road construction has developed in areas of size {new_roads_area} pixels."
    else:
        description = "No new road construction detected."

    if removed_roads_area > 0:
        description += f" Some roads have been removed or modified, covering {removed_roads_area} pixels."

    return description

# Gradio UI
demo = gr.Interface(
    fn=process_and_compare,
    inputs=[
        gr.Image(label="Upload Image 1"),
        gr.Image(label="Upload Image 2"),
        gr.Textbox(label="Coordinates of Image 1 (format: lat,lon)"),
        gr.Textbox(label="Coordinates of Image 2 (format: lat,lon)"),
        gr.Textbox(label="Altitude of Image 1 (in meters)"),
        gr.Textbox(label="Altitude of Image 2 (in meters)"),
        gr.Textbox(label="Timestamp for Image 1 (YYYY-MM-DD HH:MM:SS)"),
        gr.Textbox(label="Timestamp for Image 2 (YYYY-MM-DD HH:MM:SS)")
    ],
    outputs=[
        gr.Textbox(label="Comparison Results"),
        gr.Image(label="Segmented Image 1"),
        gr.Image(label="Segmented Image 2")
    ],
    title="DroneVision",
    description="Upload two drone images with coordinates, altitude, and timestamp. The system will validate metadata and detect road development using a U-Net model."
)

if __name__ == "__main__":
    demo.launch()

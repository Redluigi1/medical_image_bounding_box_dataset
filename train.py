import os
import json
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from glob import glob
from ultralytics import YOLO


DATA_SOURCES = [
    ("raw_images", "annotations")
    
]

DATASET_ROOT = "yolo_dataset"
CLASSES = ["target_object"]
AUG_FACTOR = 3  


transform = A.Compose(
    [
        A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Affine(scale=(0.7, 1.4), translate_percent=(0.02, 0.05), p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
        A.ImageCompression(quality_range=(30, 80), p=0.5),
        A.Downscale(scale_range=(0.5, 0.85), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
        A.HorizontalFlip(p=0.5),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

def convert_to_yolo(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x_coords, y_coords = [p[0] for p in box], [p[1] for p in box]
    xmin, xmax = max(0, min(x_coords)), min(size[0], max(x_coords))
    ymin, ymax = max(0, min(y_coords)), min(size[1], max(y_coords))
    return ((xmin + xmax) / 2.0 * dw, (ymin + ymax) / 2.0 * dh, (xmax - xmin) * dw, (ymax - ymin) * dh)

def prepare_data():
  
    for path in ["train/images", "train/labels", "val/images", "val/labels"]:
        os.makedirs(os.path.join(DATASET_ROOT, path), exist_ok=True)

    processed_data = []
    total_raw_jsons = 0

    # Iterate through all data source pairs
    for img_dir, annot_dir in DATA_SOURCES:
        json_files = glob(os.path.join(annot_dir, "*.json"))
        total_raw_jsons += len(json_files)
        
        print(f"Processing {len(json_files)} files from {annot_dir}...")

        for j_path in json_files:
            with open(j_path, "r") as f:
                data = json.load(f)

            img_path = os.path.join(img_dir, data["filename"])
            image = cv2.imread(img_path)

            if image is None:
                print(f"Warning: Image {img_path} not found. Skipping.")
                continue

            h, w, _ = image.shape
            yolo_bboxes = [list(convert_to_yolo((w, h), box)) for box in data["boxes"]]

            # 1. Original Image
            processed_data.append({
                "img": image,
                "bboxes": yolo_bboxes,
                "name": f"{os.path.basename(img_dir)}_{data['filename']}", # Unique name prefix
            })

            # 2. Augmented Images
            for i in range(AUG_FACTOR):
                augmented = transform(image=image, bboxes=yolo_bboxes, class_labels=[0] * len(yolo_bboxes))
                if len(augmented["bboxes"]) > 0:
                    processed_data.append({
                        "img": augmented["image"],
                        "bboxes": augmented["bboxes"],
                        "name": f"aug{i}_{os.path.basename(img_dir)}_{data['filename']}",
                    })

    print(f"\nSummary:")
    print(f"Total raw JSONs found: {total_raw_jsons}")
    print(f"Total images generated (with {AUG_FACTOR}x aug): {len(processed_data)}")

    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * 0.8)
    train_data, val_data = processed_data[:split_idx], processed_data[split_idx:]

    def save_subset(subset, folder):
        for item in subset:
            img_out_path = os.path.join(DATASET_ROOT, folder, "images", item["name"])
            cv2.imwrite(img_out_path, item["img"])
            
            label_name = os.path.splitext(item["name"])[0] + ".txt"
            label_out_path = os.path.join(DATASET_ROOT, folder, "labels", label_name)

            with open(label_out_path, "w") as f:
                for bbox in item["bboxes"]:
                    f.write(f"0 {' '.join(f'{c:.6f}' for c in bbox)}\n")

    save_subset(train_data, "train")
    save_subset(val_data, "val")
    print(f"Dataset preparation complete. Files saved to {DATASET_ROOT}")

def visualize_sample():
    img_list = glob(os.path.join(DATASET_ROOT, "train/images/*.jpg"))
    if not img_list: return
    
    sample_path = random.choice(img_list)
    label_path = sample_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"

    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                _, x, y, bw, bh = map(float, line.split())
                x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
                x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title(f"Check: {os.path.basename(sample_path)}")
    plt.axis("off")
    plt.show()


prepare_data()
visualize_sample()



with open("dataset.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(DATASET_ROOT)}\ntrain: train/images\nval: val/images\nnames:\n  0: target\n")


model = YOLO('yolov8s.pt')

model.train(data='dataset.yaml', epochs=15, imgsz=640, batch=16, augment=False)
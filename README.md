#  Medical Image Bounding Box Dataset

A manually annotated dataset of ~300 images with bounding boxes around **medical images only** (X-rays, MRIs, CT scans, dermatology images, etc.) designed for fine-tuning YOLO object detection models. The dataset includes both positive samples (pages with medical images) and negative samples (pages without medical images or with non-medical content).

##  Dataset Overview

| Component | Count | Description |
|-----------|-------|-------------|
| **Annotated Images** | 301 | PDF page renders with medical image annotations |

##  Purpose

This dataset was created to train a YOLO model to:
- **Detect medical images** (X-rays, CT scans, MRIs, ultrasounds, dermatology photos, etc.) within PDF documents
- **Ignore non-medical content** like charts, diagrams, logos, or decorative images
- Handle real-world variations in image placement, rotation, and document layouts

##  Dataset Structure

```
fine_tune_yolo/
├── raw_images/                 # 301 images (rendered PDF pages)
│   └── {timestamp}_{page}.jpg
├── annotations/                # Bounding box annotations (JSON)
│   └── {timestamp}_{page}.jpg.json
├── medical_images/             # Source medical images (for synthetic PDFs)
│   └── img{1-60}.jpg
├── non_medical_images/         # Source non-medical images
│   └── *.jpeg/png
├── runs/                       # Fine-tuned YOLO model weights
│   └── detect/train/weights/best.pt
├── streamlit_interface.py      # Annotation tool
├── train.py                    # Training script
├── pdf_to_img.py               # PDF to image converter
├── random_pdf_generator.tex    # LaTeX template for synthetic PDFs
└── dataset.yaml                # YOLO dataset configuration
```

##  Annotation Format



The image files are in /raw_images and the corresponding annotations in /annotations.

Annotations are stored as JSON files with the following structure:

```json
{
    "filename": "20260123_145550_100.jpg",
    "boxes": [
        [
            [453, 381],   // Top-left corner
            [760, 378],   // Top-right corner
            [456, 786],   // Bottom-left corner
            [768, 788]    // Bottom-right corner
        ],
        [
            ...
        ]
        
    ]
}
```

Each box is defined by **4 corner points** (allowing for slight rotation/skew in annotations). Empty `boxes: []` indicates no medical images on that page (negative sample).

##  Data Sources

### Medical Image Sources
- **Public Medical Journals**: Web-scraped from publicly available medical papers including:
  - UC eScholarship medical publications
  - Radiographics journal articles
  - Various case study publications (dermatology, radiology, etc.)
  - PMC open-access medical datasets

### Synthetic Data Generation
To include both positive and negative samples, synthetic PDFs were generated using:
1. **LaTeX Template** (`random_pdf_generator.tex`): Creates realistic medical report-style documents
2. **Random Image Selection** (`random_images.py`): Mixes medical images with non-medical images
3. **Layout Variations**: Full-page images, wrapped text, tables, tilted scans, etc.

##  Using the Streamlit Annotation Tool

The annotation interface allows quick bounding box creation by clicking 4 corners:

### Installation

```bash
pip install streamlit streamlit-image-coordinates pillow
```

### Running the Tool

```bash
streamlit run streamlit_interface.py
```

### Features
- **4-Point Box Drawing**: Click 4 corners to define a quadrilateral bounding box
- **Multiple Boxes**: Add multiple boxes per image
- **Jump to Image**: Navigate directly to any image number
- **Skip/Save**: Skip images without annotations or save and proceed

### Workflow
1. Click 4 points on the image to draw a rectangle around a medical image
2. Click "Add This Box" to commit the box
3. Repeat for all medical images on the page
4. Click "Save & Next" to save and move to next image
5. Use "Skip Image" for pages without medical images

##  Training YOLO

### Prerequisites

```bash
pip install ultralytics opencv-python albumentations matplotlib
```

### Running Training

```bash
python train.py
```

The training script:
1. Converts 4-point annotations to YOLO format (center-x, center-y, width, height)
2. Applies data augmentation (rotation, scaling, blur, noise, etc.)
3. Splits data 80/20 for train/validation
4. Fine-tunes YOLOv8s for 15 epochs

### Configuration

Edit `train.py` to adjust:
- `AUG_FACTOR`: Number of augmented copies per image (default: 3)
- `CLASSES`: Detection class names
- Model architecture (`yolov8s.pt`, `yolov8m.pt`, etc.)
- Training hyperparameters (epochs, batch size, image size)




##  Creating Your Own Dataset

1. **Collect PDFs**: Place documents in `docs/` folder
2. **Convert to Images**: Run `pdf_to_img.py` to extract pages
3. **Annotate**: Use `streamlit_interface.py` to draw bounding boxes
4. **Train**: Run `train.py` to fine-tune YOLO





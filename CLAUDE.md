# SkySense

Drone-view object detection using YOLOv8 trained on the VisDrone dataset.

## Target Classes

| ID | Class     | VisDrone source classes      |
|----|-----------|------------------------------|
| 0  | person    | pedestrian (0) + people (1)  |
| 1  | car       | car (3)                      |
| 2  | motorbike | motor (5)                    |
| 3  | truck     | truck (9)                    |

VisDrone has 10 classes total. We filter and remap to these 4.

## Tech Stack

- **Model**: YOLOv8 (Ultralytics) — pretrained on COCO, fine-tuned on VisDrone
- **Dataset**: VisDrone (auto-downloaded via Ultralytics)
- **Runtime**: Google Colab with GPU (T4 for free tier)
- **Framework**: PyTorch (via Ultralytics)
- **Export**: ONNX for deployment

## Repository Structure

```
skysense/
  CLAUDE.md                          # This file
  skysense_drone_detection.ipynb     # Main training notebook (run in Colab)
```

## Notebook Design

The notebook (`skysense_drone_detection.ipynb`) has a **configuration cell** at the top:

```python
MODEL_SIZE = "yolov8s"   # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16          # Reduce for larger models
```

Pipeline: Install → Download VisDrone → Remap classes → Train → Evaluate → Visualize → Export

## Scaling

| Model    | Params | Free Colab (T4) | Notes            |
|----------|--------|------------------|------------------|
| yolov8n  | 3.2M   | Yes              | Fastest          |
| yolov8s  | 11.2M  | Yes (default)    | Good balance     |
| yolov8m  | 25.9M  | Yes (batch ≤ 8)  | Higher accuracy  |
| yolov8l  | 43.7M  | Tight            | Needs low batch  |
| yolov8x  | 68.2M  | Colab Pro / A100 | Best accuracy    |

## Usage

1. Open `skysense_drone_detection.ipynb` in Google Colab
2. Set runtime to **GPU** (`Runtime > Change runtime type > T4 GPU`)
3. Adjust `MODEL_SIZE` / `BATCH_SIZE` / `EPOCHS` in the config cell if needed
4. Run all cells sequentially
5. Download the trained weights (`best.pt`) and ONNX export from the final cell

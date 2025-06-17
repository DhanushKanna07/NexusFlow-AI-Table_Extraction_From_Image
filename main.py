import torch
import pytesseract
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import logging
import re
import os
from typing import List, Dict, Tuple
from shapely.geometry import box as shapely_box

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load image
image_path = r"D:\\projects\\nexusai\\table_extractor\\sample_images\\med_2.png"
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    logging.error(f"Failed to load image: {e}")
    raise
width, height = image.size

# Load model
model_name = "microsoft/table-transformer-structure-recognition"
try:
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)
except Exception as e:
    logging.error(f"Failed to load model or processor: {e}")
    raise

# Run detection
encoding = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**encoding)

label_thresholds = {
    "table row": 0.55,
    "table column": 0.65,
    "table column header": 0.7,
    "table spanning cell": 0.40
}

results = processor.post_process_object_detection(
    outputs, threshold=min(label_thresholds.values()), target_sizes=[(height, width)]
)[0]

# Enhancement for low-quality OCR
def enhance_low_quality_image(crop: Image.Image) -> Image.Image:
    crop = crop.convert("L")
    crop = ImageOps.invert(crop)
    crop = ImageEnhance.Contrast(crop).enhance(2.5)
    crop = ImageEnhance.Sharpness(crop).enhance(2.0)
    crop = crop.filter(ImageFilter.MedianFilter(size=3))
    return crop

# OCR cleaning
def clean_ocr_text(text: str) -> str:
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"[|*\"\u00a2'()\[\]{}®]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def extract_ocr_text(crop: Image.Image) -> str:
    try:
        crop = enhance_low_quality_image(crop)
        text = pytesseract.image_to_string(crop, config="--psm 6 --oem 3")
        text = re.sub(r"\n+", " ", text)
        return clean_ocr_text(text)
    except Exception as e:
        logging.warning(f"OCR failed: {e}")
        return ""

# Snap logic
def snap_to_nearest(pixel: int, anchors: List[int], threshold: int = 8) -> int:
    for a in anchors:
        if abs(pixel - a) < threshold:
            return a
    return pixel

# Prepare snapping anchors
initial_row_anchors, initial_col_anchors = [], []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    label_name = model.config.id2label[int(label)]
    if label_name not in label_thresholds or score < label_thresholds[label_name]:
        continue
    box = list(map(int, box.tolist()))
    if label_name == "table row":
        initial_row_anchors.append((box[1] + box[3]) // 2)
    elif label_name == "table column":
        initial_col_anchors.append((box[0] + box[2]) // 2)

# Collect and snap boxes
row_boxes, col_boxes, spanning_cells = [], [], []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    label_name = model.config.id2label[int(label)]
    if label_name not in label_thresholds or score < label_thresholds[label_name]:
        continue
    box = list(map(int, box.tolist()))
    box[0] = snap_to_nearest(box[0], initial_col_anchors)
    box[2] = snap_to_nearest(box[2], initial_col_anchors)
    box[1] = snap_to_nearest(box[1], initial_row_anchors)
    box[3] = snap_to_nearest(box[3], initial_row_anchors)
    if label_name == "table row":
        row_boxes.append(box)
    elif label_name == "table column":
        col_boxes.append(box)
    elif label_name == "table spanning cell":
        spanning_cells.append(box)

# IoU deduplication
def iou(boxA, boxB):
    bA = shapely_box(*boxA)
    bB = shapely_box(*boxB)
    return bA.intersection(bB).area / bA.union(bB).area

def remove_duplicate_boxes(boxes, iou_threshold=0.85):
    final = []
    for b in boxes:
        if all(iou(b, f) < iou_threshold for f in final):
            final.append(b)
    return final

row_boxes = remove_duplicate_boxes(row_boxes)
col_boxes = remove_duplicate_boxes(col_boxes)
row_boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
col_boxes.sort(key=lambda b: (b[0] + b[2]) / 2)

# Spanning logic
def is_within_spanning_cell(x0, y0, x1, y1, spanning_cells):
    for span_box in spanning_cells:
        sx0, sy0, sx1, sy1 = span_box
        if x0 >= sx0 and x1 <= sx1 and y0 >= sy0 and y1 <= sy1:
            return True, span_box, [sx0, sy0, sx1, sy1]
    return False, None, None

# Merge rows
def merge_cells_in_row(row, col_boxes, spanning_cells, image):
    merged_row = []
    col_idx = 0
    while col_idx < len(row):
        x0, y0 = col_boxes[col_idx][0], col_boxes[col_idx][1]
        x1, y1 = col_boxes[col_idx][2], col_boxes[col_idx][3]
        is_spanning, span_box, span_coords = is_within_spanning_cell(x0, y0, x1, y1, spanning_cells)

        if is_spanning:
            crop = image.crop(span_coords)
            text = extract_ocr_text(crop)
            merged_row.append(text)
            while col_idx + 1 < len(row) and col_boxes[col_idx + 1][0] < span_coords[2]:
                col_idx += 1
        else:
            if col_idx + 1 < len(row) and "pere" in row[col_idx + 1].lower():
                combined_text = f"{row[col_idx]} {row[col_idx + 1].replace('pere', 'Ampere')}"
                merged_row.append(clean_ocr_text(combined_text))
                col_idx += 1
            else:
                merged_row.append(row[col_idx])
        col_idx += 1
    return merged_row

# Extract grid
grid = []
spanning_cell_texts = {}
for row_box in row_boxes:
    row = []
    for col_box in col_boxes:
        x0 = max(row_box[0], col_box[0])
        y0 = max(row_box[1], col_box[1])
        x1 = min(row_box[2], col_box[2])
        y1 = min(row_box[3], col_box[3])
        if x1 - x0 < 4 or y1 - y0 < 4:
            row.append("")
            continue
        is_spanning, span_box, span_coords = is_within_spanning_cell(x0, y0, x1, y1, spanning_cells)
        if is_spanning:
            if tuple(span_box) not in spanning_cell_texts:
                crop = image.crop(span_box)
                text = extract_ocr_text(crop)
                spanning_cell_texts[tuple(span_box)] = text
            row.append(spanning_cell_texts[tuple(span_box)])
        else:
            crop = image.crop((x0, y0, x1, y1))
            text = extract_ocr_text(crop)
            row.append(text)
    grid.append(merge_cells_in_row(row, col_boxes, spanning_cells, image))

# Save output
def convert_to_dict(grid: List[List[str]]) -> Dict[str, List[str]]:
    return {f"row_{i:02}": row for i, row in enumerate(grid)}

final_table = convert_to_dict(grid)
output_dir = "table_output"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "output.json"), "w", encoding="utf-8") as f:
    json.dump(final_table, f, indent=2, ensure_ascii=False)

# Visualize
plt.figure(figsize=(16, 10))
plt.imshow(image)
ax = plt.gca()
for row_box in row_boxes:
    for col_box in col_boxes:
        x0 = max(row_box[0], col_box[0])
        y0 = max(row_box[1], col_box[1])
        x1 = min(row_box[2], col_box[2])
        y1 = min(row_box[3], col_box[3])
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='green', linewidth=1))
for span_box in spanning_cells:
    x0, y0, x1, y1 = span_box
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', linewidth=2))
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "table_marked.png"), dpi=300)
plt.close()

print("✅ Marked table image saved at:", os.path.join(output_dir, "table_marked.png"))
print("✅ Table data saved at:", os.path.join(output_dir, "output.json"))

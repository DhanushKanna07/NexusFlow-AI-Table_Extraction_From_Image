import os
import fitz  # PyMuPDF
import torch
import pytesseract
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from shapely.geometry import box as shapely_box
import re
import logging
from typing import List, Dict, Tuple


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TableDetector:
    def __init__(self, pdf_path: str, output_root: str, zoom: int = 3, min_table_threshold: float = 0.45, iou_merge_threshold: float = 0.5):
        self.pdf_path = pdf_path
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)
        self.zoom = zoom
        self.min_table_threshold = min_table_threshold
        self.iou_merge_threshold = iou_merge_threshold
        logging.info("🔍 Loading detection model...")
        self.det_proc = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.det_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    def convert_pdf_to_images(self) -> List[Tuple[int, Image.Image]]:
        doc = fitz.open(self.pdf_path)
        images = []
        for page_num in range(len(doc)):
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = doc.load_page(page_num).get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_num + 1, img))
        doc.close()
        return images

    def is_valid_table_shape(self, box: List[int], min_ar: float = 0.5, max_ar: float = 5.0, min_area: int = 5000) -> bool:
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0
        if height == 0 or width == 0:
            return False
        ar = width / height
        area = width * height
        return min_ar <= ar <= max_ar and area >= min_area

    def is_text_dense_enough(self, image: Image.Image, box: List[int], min_lines: int = 2, min_words: int = 3) -> bool:
        crop = image.crop(box)
        text = pytesseract.image_to_string(crop, config="--psm 6")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        word_count = sum(len(l.split()) for l in lines)
        return len(lines) >= min_lines and word_count >= min_words

    def is_not_a_real_table(self, image: Image.Image, box: List[int]) -> bool:
        crop = image.crop(box)
        text = pytesseract.image_to_string(crop, config="--psm 6")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return True
        word_counts = [len(l.split()) for l in lines]
        avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0
        wc_std = np.std(word_counts) if word_counts else 0
        unique_words = set(word for l in lines for word in l.split())
        is_code_like = all(len(set(l)) < 15 for l in lines[:4])
        is_bullet_like = sum(l.startswith(("-", "*", "•")) for l in lines) > len(lines) * 0.5
        high_repetition = len(unique_words) < avg_wc * 1.5
        bullet_or_code = is_bullet_like or is_code_like
        extremely_uniform = wc_std < 1 and avg_wc > 6
        short_and_repetitive = avg_wc < 3 and high_repetition
        #header_miss = not self.looks_like_table_header(lines[0]) if lines else True
        #return bullet_or_code or extremely_uniform or short_and_repetitive or header_miss
        return bullet_or_code or extremely_uniform or short_and_repetitive



    def score_table_confidence(self, image: Image.Image, box: List[int]) -> int:
        crop = image.crop(box)
        text = pytesseract.image_to_string(crop, config="--psm 6")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        word_counts = [len(l.split()) for l in lines]
        avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0
        wc_std = np.std(word_counts) if word_counts else 0
        unique_words = set(word for l in lines for word in l.split())

        score = 0
        if len(lines) >= 2 and avg_wc >= 2:
            score += 1
        if wc_std >= 1.0:
            score += 1
        if len([l for l in lines if len(l.split()) > 6]) < len(lines) * 0.7:
            score += 1
        if all(len(set(l)) > 10 for l in lines[:3]):
            score += 1
        if self.has_clear_grid_lines(crop):
            score += 1  # New: bonus for grid-based layout
        if lines and self.looks_like_table_header(lines[0]):
            score += 1  # New: bonus for header keyword match
        return score

    def merge_overlapping_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        merged = []
        for box in boxes:
            new_box = shapely_box(*box)
            merged_flag = False
            for i, existing in enumerate(merged):
                iou = new_box.intersection(existing).area / new_box.union(existing).area
                if iou > self.iou_merge_threshold or new_box.distance(existing) < 40:
                    merged[i] = merged[i].union(new_box)
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(new_box)
        return [list(map(int, b.bounds)) for b in merged]
    
    def expand_box(self, box: List[int], image: Image.Image, pad_w: int = 20, pad_h: int = 10) -> List[int]:
        x0, y0, x1, y1 = box
        w, h = image.size
        return [
            max(0, x0 - pad_w),
            max(0, y0 - pad_h),
            min(w, x1 + pad_w),
            min(h, y1 + pad_h)
        ]



    def detect_tables(self, table_count: int) -> Tuple[List[Tuple[int, Image.Image, List[int], int]], int]:
        pdf_images = self.convert_pdf_to_images()
        all_tables = []
        
        for page_num, image in pdf_images:
            logging.info(f"📄 Detecting tables on Page {page_num}...")
            inputs = self.det_proc(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.det_model(**inputs)
            
            w, h = image.size
            results = self.det_proc.post_process_object_detection(outputs, threshold=self.min_table_threshold, target_sizes=[(h, w)])[0]
            raw_boxes = [
                list(map(int, box.tolist()))
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
                if self.det_model.config.id2label[label.item()] == "table" and score >= self.min_table_threshold
            ]
            
            filtered_boxes = []
            for box in raw_boxes:
                score = self.score_table_confidence(image, box)
                if (
                    self.is_valid_table_shape(box)
                    and self.is_text_dense_enough(image, box, min_lines=1, min_words=2)
                    #and not self.is_not_a_real_table(image, box)
                    and score >= 3  # Accepts wider range of valid-looking tables
                ):
                    logging.info(f"✅ Accepted box {box} with score {score}")
                    filtered_boxes.append(box)
                else:
                    logging.info(f"❌ Rejected box {box} with score {score}")

            
            final_boxes = self.merge_overlapping_boxes(filtered_boxes)
            
            for box in final_boxes:
                table_count += 1
                expanded_box = self.expand_box(box, image)
                cropped = image.crop(expanded_box)
                padded = self.pad_image(cropped)
                crop_fname = f"image_{table_count}_cropped.png"
                padded.save(os.path.join(self.output_root, crop_fname))
                logging.info(f"💾 Saved cropped table image: {crop_fname}")
                all_tables.append((page_num, padded, box, table_count))
        
        return all_tables, table_count

    def pad_image(self, image: Image.Image, padding_px: int = 80) -> Image.Image:
        w, h = image.size
        new_img = Image.new("RGB", (w + 2*padding_px, h + 2*padding_px), "white")
        new_img.paste(image, (padding_px, padding_px))
        return new_img
    
    
    def has_clear_grid_lines(self, image: Image.Image, threshold: float = 0.1) -> bool:
        np_img = np.array(image.convert("L"))
        horizontal_edges = np.sum(np.abs(np.diff(np_img, axis=1)) > 30, axis=1)
        vertical_edges = np.sum(np.abs(np.diff(np_img, axis=0)) > 30, axis=0)
        h_density = np.sum(horizontal_edges > np.mean(horizontal_edges)) / len(horizontal_edges)
        v_density = np.sum(vertical_edges > np.mean(vertical_edges)) / len(vertical_edges)
        return h_density > threshold and v_density > threshold


    def looks_like_table_header(self, text: str) -> bool:
        keywords = ["qty", "quantity", "rate", "amount", "unit", "description", "total", "cost", "sr.", "no.", "price", "charge", "date", "item"]
        text = text.lower()
        return any(kw in text for kw in keywords)



class TableStructureRecognizer:
    def __init__(self, output_root: str, snap_threshold_scale: float = 0.015):
        self.output_root = output_root
        self.snap_threshold_scale = snap_threshold_scale
        logging.info("🔍 Loading structure recognition model...")
        self.struct_proc = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.struct_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    def enhance_low_quality_image(self, crop: Image.Image) -> Image.Image:
        # crop = crop.convert("L")
        # crop = ImageOps.invert(crop)
        # crop = ImageEnhance.Contrast(crop).enhance(2.5)
        # crop = ImageEnhance.Sharpness(crop).enhance(2.0)
        # crop = crop.filter(ImageFilter.MedianFilter(size=3))
        # return crop
        crop = crop.convert("L")  # Grayscale
        crop = ImageOps.autocontrast(crop)
        crop = ImageEnhance.Contrast(crop).enhance(1.8)
        crop = ImageEnhance.Sharpness(crop).enhance(2.5)
        crop = crop.filter(ImageFilter.MedianFilter(size=3))
        return crop

    def clean_ocr_text(self, text: str) -> str:
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
        text = re.sub(r"[|*\"\u00a2'()\[\]{}®]", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def extract_ocr_text(self, crop: Image.Image) -> str:
        try:
            # crop = self.enhance_low_quality_image(crop)
            # text = pytesseract.image_to_string(crop, config="--psm 6 --oem 3")
            # text = re.sub(r"\n+", " ", text)
            # return self.clean_ocr_text(text)
            crop = self.enhance_low_quality_image(crop)

            w, h = crop.size
            aspect_ratio = w / h if h != 0 else 1
            psm_mode = "7" if aspect_ratio > 2.5 or w < 120 else "6"
            config = f"--psm {psm_mode} --oem 3"
            text = pytesseract.image_to_string(crop, config=config)
            text = re.sub(r"\n+", " ", text)
            return self.clean_ocr_text(text)
        except Exception as e:
            logging.warning(f"OCR failed: {e}")
            return ""

    def snap_to_nearest(self, pixel: int, anchors: List[int], threshold: int) -> int:
        if not anchors:
            return pixel
        closest = min(anchors, key=lambda a: abs(a - pixel))
        if abs(pixel - closest) <= threshold:
            return closest
        return pixel

    def iou(self, boxA: List[int], boxB: List[int]) -> float:
        bA = shapely_box(*boxA)
        bB = shapely_box(*boxB)
        return bA.intersection(bB).area / bA.union(bB).area

    def remove_duplicate_boxes(self, boxes: List[List[int]], iou_threshold: float = 0.98) -> List[List[int]]:
        final = []
        for b in boxes:
            if not final:
                final.append(b)
                continue
            is_duplicate = False
            for f in final:
                iou_val = self.iou(b, f)
                b_area = (b[2] - b[0]) * (b[3] - b[1])
                f_area = (f[2] - f[0]) * (f[3] - f[1])
                area_ratio = min(b_area, f_area) / max(b_area, f_area)
                if iou_val > iou_threshold and area_ratio > 0.85:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final.append(b)
        return final

    def is_within_spanning_cell(self, x0: int, y0: int, x1: int, y1: int, spanning_cells: List[List[int]]) -> Tuple[bool, List[int], List[int]]:
        for span_box in spanning_cells:
            sx0, sy0, sx1, sy1 = span_box
            if x0 >= sx0 and x1 <= sx1 and y0 >= sy0 and y1 <= sy1:
                return True, span_box, [sx0, sy0, sx1, sy1]
        return False, None, None

    def merge_cells_in_row(self, row: List[str], col_boxes: List[List[int]], spanning_cells: List[List[int]], image: Image.Image) -> List[str]:
        merged_row = []
        col_idx = 0
        while col_idx < len(row):
            x0, y0 = col_boxes[col_idx][0], col_boxes[col_idx][1]
            x1, y1 = col_boxes[col_idx][2], col_boxes[col_idx][3]
            is_spanning, span_box, span_coords = self.is_within_spanning_cell(x0, y0, x1, y1, spanning_cells)
            if is_spanning:
                crop = image.crop(span_coords)
                text = self.extract_ocr_text(crop)
                merged_row.append(text)
                while col_idx + 1 < len(row) and col_boxes[col_idx + 1][0] < span_coords[2]:
                    col_idx += 1
            else:
                if col_idx + 1 < len(row) and "pere" in row[col_idx + 1].lower():
                    combined_text = f"{row[col_idx]} {row[col_idx + 1].replace('pere', 'Ampere')}"
                    merged_row.append(self.clean_ocr_text(combined_text))
                    col_idx += 1
                else:
                    merged_row.append(row[col_idx])
            col_idx += 1
        return merged_row

    def draw_table_structure(self, image: Image.Image, row_boxes: List[List[int]], col_boxes: List[List[int]], spanning_cells: List[List[int]]) -> Image.Image:
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        for row_box in row_boxes:
            for col_box in col_boxes:
                x0 = max(row_box[0], col_box[0])
                y0 = max(row_box[1], col_box[1])
                x1 = min(row_box[2], col_box[2])
                y1 = min(row_box[3], col_box[3])
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                draw.rectangle((x0, y0, x1, y1), outline="green", width=1)
        for span_box in spanning_cells:
            draw.rectangle(span_box, outline="red", width=2)
        return img_draw
    
    def detect_alignment(self, image: Image.Image) -> str:
        """Detect horizontal text alignment: left, center, or right."""
        np_img = np.array(image.convert("L"))
        threshold = 180
        binary = np.where(np_img < threshold, 1, 0)

        col_sums = binary.sum(axis=0)
        total = col_sums.sum()

        if total == 0:
            return "unknown"

        left = col_sums[:len(col_sums)//3].sum()
        center = col_sums[len(col_sums)//3:2*len(col_sums)//3].sum()
        right = col_sums[2*len(col_sums)//3:].sum()

        max_zone = max(left, center, right)
        if max_zone == left:
            return "left"
        elif max_zone == center:
            return "center"
        else:
            return "right"


    def extract_table_structure(self, image: Image.Image, box: List[int], page_num: int, table_idx: int) -> Tuple[Dict[str, List[str]], Image.Image]:
        w, h = image.size
        encoding = self.struct_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.struct_model(**encoding)
        
        table_area = w * h
        label_thresholds = {
            "table row": max(0.50, 0.55 - (table_area / 1e6)),
            "table column": max(0.60, 0.65 - (table_area / 1e6)),
            "table column header": max(0.65, 0.70 - (table_area / 1e6)),
            "table spanning cell": max(0.35, 0.40 - (table_area / 1e6))
        }
        
        results = self.struct_proc.post_process_object_detection(
            outputs, threshold=min(label_thresholds.values()), target_sizes=[(h, w)]
        )[0]
        
        initial_row_anchors, initial_col_anchors = [], []
        anchor_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.struct_model.config.id2label[int(label)]
            if label_name not in label_thresholds or score < label_thresholds[label_name]:
                continue
            box = list(map(int, box.tolist()))
            anchor_boxes.append((score, label_name, box))
            if label_name == "table row":
                initial_row_anchors.append(box[1])
                initial_row_anchors.append(box[3])
            elif label_name == "table column":
                initial_col_anchors.append(box[0])
                initial_col_anchors.append(box[2])
        
        initial_row_anchors = sorted(list(set(initial_row_anchors)))
        initial_col_anchors = sorted(list(set(initial_col_anchors)))
        
        snap_threshold = int(max(5, min(w, h) * self.snap_threshold_scale))
        
        row_boxes, col_boxes, spanning_cells = [], [], []
        for score, label_name, box in anchor_boxes:
            if label_name not in label_thresholds or score < label_thresholds[label_name]:
                continue
            box = list(map(int, box))
            box[0] = self.snap_to_nearest(box[0], initial_col_anchors, snap_threshold)
            box[2] = self.snap_to_nearest(box[2], initial_col_anchors, snap_threshold)
            box[1] = self.snap_to_nearest(box[1], initial_row_anchors, snap_threshold)
            box[3] = self.snap_to_nearest(box[3], initial_row_anchors, snap_threshold)
            if label_name == "table row":
                row_boxes.append(box)
            elif label_name == "table column":
                col_boxes.append(box)
            elif label_name == "table spanning cell":
                spanning_cells.append(box)
        
        row_boxes = self.remove_duplicate_boxes(row_boxes)
        col_boxes = self.remove_duplicate_boxes(col_boxes)
        row_boxes.sort(key=lambda b: b[1])
        col_boxes.sort(key=lambda b: b[0])
        
        if not row_boxes or not col_boxes:
            logging.warning("No valid rows or columns detected. Returning empty table.")
            return {}, image
        
        grid = []
        spanning_cell_texts = {}

        for row_box in row_boxes:
            row = []
            for col_box in col_boxes:
                x0 = max(row_box[0], col_box[0])
                y0 = max(row_box[1], col_box[1])
                x1 = min(row_box[2], col_box[2])
                y1 = min(row_box[3], col_box[3])
                if x1 - x0 < 2 or y1 - y0 < 2:
                    row.append({"text": "", "align": "unknown"})
                    continue

                is_spanning, span_box, span_coords = self.is_within_spanning_cell(x0, y0, x1, y1, spanning_cells)
                if is_spanning:
                    span_key = tuple(span_box)
                    if span_key not in spanning_cell_texts:
                        crop = image.crop(span_box)
                        text = self.extract_ocr_text(crop)
                        spanning_cell_texts[span_key] = text
                    row.append({
                        "text": spanning_cell_texts[span_key],
                        "align": self.detect_alignment(image.crop(span_coords))
                    })
                else:
                    crop = image.crop((x0, y0, x1, y1))
                    text = self.extract_ocr_text(crop)
                    row.append({
                        "text": text,
                        "align": self.detect_alignment(crop)
                    })
            grid.append(row)
            #grid.append(self.merge_cells_in_row(row, col_boxes, spanning_cells, image))
        boxed_image = self.draw_table_structure(image, row_boxes, col_boxes, spanning_cells)
        return {f"row_{i:02}": row for i, row in enumerate(grid)}, boxed_image

# --- Main Processing Flow ---
def main():
    # Define pdf_path and output_root
    pdf_path = r"D:\projects\nexusai\table_extractor\table_test.pdf"
    output_root = "extracted_tables"
    
    detector = TableDetector(pdf_path=pdf_path, output_root=output_root)
    recognizer = TableStructureRecognizer(output_root=output_root)
    
    table_count = 0
    tables, table_count = detector.detect_tables(table_count)
    
    for page_num, cropped_image, box, table_idx in tables:
        table_data, boxed_image = recognizer.extract_table_structure(cropped_image, box, page_num, table_idx)
        
        boxed_fname = f"image_{table_idx}_boxed.png"
        boxed_image.save(os.path.join(output_root, boxed_fname))
        logging.info(f"💾 Saved boxed table image: {boxed_fname}")
        
        table_json_data = {
            "page": page_num,
            "boxed_image": boxed_fname,
            "data": table_data
        }
        
        table_json_filename = f"image_{table_idx}.json"
        table_json_path = os.path.join(output_root, table_json_filename)
        
        with open(table_json_path, "w", encoding="utf-8") as jf:
            json.dump(table_json_data, jf, indent=2, ensure_ascii=False)
        
        logging.info(f"📄 JSON saved for Table {table_idx}: {table_json_filename}")

if __name__ == "__main__":
    main()

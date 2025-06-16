#  Table Extractor using Table Transformer + OCR

This project extracts structured table data from images using [Microsoft's Table Transformer (DETR-based)](https://huggingface.co/microsoft/table-transformer-structure-recognition) and `pytesseract` OCR. It supports cell-level extraction, even for complex tables with precise grid mapping.

---

##  Features

-  Deep learning-based **table structure recognition**
-  Cell-by-cell **OCR extraction**
-  Cell-grid visualization with bounding boxes
-  Outputs JSON with structured table rows
-  Saves marked-up table image for visual verification

---


---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt


Make sure tesseract is available in your system PATH. If not, set the path manually in Python:


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


How It Works
Uses the microsoft/table-transformer-structure-recognition model to detect:

Table rows

Table columns

Intersects row and column bounding boxes to get each cell.

Performs OCR (pytesseract) on every cell.

Saves the table as JSON + overlays the detected grid on the original image.

# LICENSE (MIT)



Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      
copies of the Software, and to permit persons to whom the Software is          
furnished to do so, subject to the following conditions:                       

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.                                

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.



import fitz  # PyMuPDF
import numpy as np
from pdf2image import convert_from_path
import pytesseract

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to make the text more visible
    _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert back to PIL Image format
    return Image.fromarray(thresh)

def pdf_to_images(pdf_path):
    # Convert PDF to images
    return convert_from_path(pdf_path)

def extract_text_from_image(image):
    # Use Tesseract OCR to extract text from the image
    return pytesseract.image_to_string(preprocess_image(image), config='--psm 6')

def extract_handwritten_text_from_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    text = ""
    for i, image in enumerate(images):
        text += f"Page {i+1}:\n"
        text += extract_text_from_image(image) + "\n"
    return text

# Example usage
pdf_path = "/Users/samhithdara/Downloads/ALGO DS Assignment 1/EMP.pdf"
# pdf_path = "/Users/samhithdara/Downloads/homework2.pdf"
text = extract_handwritten_text_from_pdf(pdf_path)
print(text)


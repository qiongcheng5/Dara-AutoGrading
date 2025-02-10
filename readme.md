# Automated Answer Extraction and Grading System

This project automates the process of extracting answers from student submissions (PDF and DOCX files) and grading them using Retrieval-Augmented Generation (RAG) techniques combined with AI-powered assessment.

## Features

### Document Processing (Answer Extraction)
- Supports PDF, Image and DOCX formats.
- Extracts questions and answers from structured and unstructured documents.
- Uses OpenAI for handwritten text recognition.
- Stores extracted text in an organized format for grading.

### Automatic Grading
- Implements Retrieval-Augmented Generation (RAG) for intelligent grading.
- Retrieves relevant textbook material using vector-based search.
- Grades student answers by comparing them to the extracted textbook context.
- Supports multiple rubrics and confidence-based grading.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pdf_reader.git
   cd pdf_reader
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up OpenAI API and vector database credentials in `.env`.

## Usage

1. Extract answers from student submissions folder:
   ```bash
   python extract_answers.py
   ```
2. Populate vector DB with related documents for RAG:
   ```bash
   python vector_update.py
   ```
3. Process extracted answers for grading:
   ```bash
   python autograder.py
   ```
4. Evaluate results using Langfuse:
   ```bash
   python evaluate.py
   ```

## Future Enhancements
- Improve OCR accuracy for handwritten submissions.
- Optimize vector retrieval for faster grading.
- Integrate a web-based interface for result visualization.



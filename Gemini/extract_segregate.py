import os
import csv
import json
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pydantic import BaseModel, Field
import mammoth
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
import io

load_dotenv()

class GeminiImageProcessor:
    class Config(BaseModel):
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GEMINI_API_KEY"),
            description="Your Google API key for image processing",
        )

    def __init__(self):
        self.config = self.Config()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def process_image(self, image: Image.Image) -> str:
        """Process an image with Gemini and return extracted text."""
        try:
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY is required for image processing")

            if image.format != 'JPEG':
                image = image.convert('RGB')

            with io.BytesIO() as img_byte_arr:
                image.save(img_byte_arr, format="JPEG")
                image_bytes = img_byte_arr.getvalue()

            image_part = {"mime_type": "image/jpeg", "data": image_bytes}
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = "Give all the text present in this image without any extra parts."
            response = model.generate_content([prompt, image_part])
            return response.text
        except Exception as e:
            return f"[Error processing image: {str(e)}]"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        images = convert_from_path(pdf_path)
        return " ".join(self.process_image(image) for image in images)


class FileProcessor:
    def __init__(self, gemini_processor: GeminiImageProcessor):
        self.gemini_processor = gemini_processor

    def process_file(self, file_path: str) -> str:
        extension = file_path.split(".")[-1].lower()
        if extension == "html":
            return self.extract_text_from_html(file_path)
        elif extension == "pdf":
            return self.gemini_processor.extract_text_from_pdf(file_path)
        elif extension == "docx":
            return self.extract_text_from_docx_mammoth(file_path)
        elif extension in ("png", "jpg"):
            return self.gemini_processor.process_image(Image.open(file_path))
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def extract_text_from_html(self, html_path: str) -> str:
        with open(html_path, "r", encoding="utf-8") as file:
            return BeautifulSoup(file.read(), "html.parser").get_text()

    def extract_text_from_docx_mammoth(self, docx_path: str) -> str:
        with open(docx_path, "rb") as docx_file:
            return mammoth.extract_raw_text(docx_file).value

    def process_folder(self, folder_path: str, output_csv: str):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        with open(output_csv, mode='w', newline='') as csv_file, ThreadPoolExecutor(max_workers=5) as executor:
            writer = csv.writer(csv_file)
            writer.writerow(['File name', 'Extracted Text'])
            future_to_file = {executor.submit(self.process_file, os.path.join(folder_path, f)): f for f in files}
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    writer.writerow([future_to_file[future], result])


class CSVProcessor:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)

    def process_text_with_gemini(self, questions, extracted_text):
        prompt = (f"Given the answers {extracted_text} written by students for questions {questions}, "
                  f"give me what student has answered for each question without manipulating text, "
                  f"and complete text till the start of the next question, separated by `###`")
        response = self.model.generate_content(prompt)
        return response.text

    def process_csv(self, input_csv, output_csv, questions):
        with open(input_csv, newline='', encoding='utf-8') as csvfile, \
                open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(csvfile)
            question_headers = questions.split("\n\n")
            fieldnames = ["File name"] + question_headers
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                file_name = row["File name"]
                extracted_text = row["Extracted Text"]
                processed_text = self.process_text_with_gemini(questions, extracted_text)
                answers_list = processed_text.split("###")
                row_data = {"File name": file_name}
                for i, question in enumerate(question_headers):
                    row_data[question] = answers_list[i].strip() if i < len(answers_list) else ""
                writer.writerow(row_data)


def read_questions(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


if __name__ == "__main__":
    gemini_processor = GeminiImageProcessor()
    file_processor = FileProcessor(gemini_processor)
    file_processor.process_folder("/Users/samhithdara/PycharmProjects/deepseek/docs", "extracted_text.csv")
    questions = read_questions("/Users/samhithdara/PycharmProjects/deepseek/actual_questions.json")
    csv_processor = CSVProcessor()
    csv_processor.process_csv("extracted_text.csv", "segregated.csv", questions["pseudo_num"])

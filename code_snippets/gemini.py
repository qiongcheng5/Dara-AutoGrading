import io
import os
import time
import asyncio
import google.generativeai as genai

import os
import base64
import csv
import json
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pydantic import BaseModel, Field
from openai import OpenAI
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import mammoth

# Set environment variables for OpenAI API key
os.environ["OPENAI_API_KEY"] = "key"  # Replace with actual key or set via environment
api_key = os.getenv('OPENAI_API_KEY')

class TextExtractionService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR and OpenAI API."""
        images = convert_from_path(pdf_path)
        text = ''
        for image in images:
            base64_image = self.encode_image_to_base64(image)
            messages = self._create_image_messages(base64_image)
            text += self._process_openai(messages)
        return text

    def extract_text_from_html(self, html_path: str) -> str:
        """Extract text from HTML and convert it to LaTeX."""
        with open(html_path, "r", encoding="utf-8") as file:
            html_content = file.read()
            return self.convert_to_latex(html_content)

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OpenAI."""
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            messages = self._create_image_messages(base64_image)
            return self._process_openai(messages)

    def extract_student_answers(self, questions: str, answers: str) -> str:
        """Extract answers corresponding to questions."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the answers {answers} written by students for questions {questions}, "
                                f"give me what student has answered for each question without manipulating text, "
                                f"and complete text till the start of the next question, separated by `###`."
                    }
                ]
            }
        ]
        return self._process_openai(messages)

    # def _process_openai(self, messages: list) -> str:
    #     """Process messages through OpenAI API and return the result."""
    #     completion = self.client.chat.completions.create(
    #         messages=messages,
    #         model="gpt-4-turbo"
    #     )
    #     return completion.choices[0].message.content

    def _create_image_messages(self, base64_image: str) -> list:
        """Create messages for OpenAI API to extract text from an image."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given an image of handwritten student answers, give me all the text present in the image "
                                "and avoid extra lines."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

    def convert_to_latex(self, text: str) -> str:
        """Convert HTML-based math tags to LaTeX equivalents."""
        soup = BeautifulSoup(text, "html.parser")
        self._convert_html_tags_to_latex(soup)
        return soup.get_text()

    def _convert_html_tags_to_latex(self, soup):
        """Helper method to convert specific HTML tags to LaTeX."""
        for sup in soup.find_all("sup"):
            latex_sup = f"^{{{sup.get_text()}}}"
            sup.replace_with(latex_sup)
        for sub in soup.find_all("sub"):
            latex_sub = f"_{{{sub.get_text()}}}"
            sub.replace_with(latex_sub)
        for frac in soup.find_all("span", class_="frac"):
            numerator = frac.find("span", class_="num")
            denominator = frac.find("span", class_="den")
            if numerator and denominator:
                latex_frac = r"\frac{" + numerator.get_text() + "}{" + denominator.get_text() + "}"
                frac.replace_with(latex_frac)
        for sqrt in soup.find_all("span", class_="sqrt"):
            latex_sqrt = r"\sqrt{" + sqrt.get_text() + "}"
            sqrt.replace_with(latex_sqrt)

class FileProcessor:
    def __init__(self, text_extractor: TextExtractionService):
        self.text_extractor = text_extractor

    def extract_name_and_id(self, file_name: str) -> tuple:
        """Extract student name and ID from file name."""
        name_id = os.path.splitext(file_name)[0]
        splits = name_id.split('_')
        return splits[1], splits[0]  # Assuming format: student_id_name

    def process_file(self, file_path: str) -> str:
        """Process the file based on its extension."""
        extension = file_path.split(".")[-1].lower()

        if extension == "html":
            return self.text_extractor.extract_text_from_html(file_path)
        elif extension == "pdf":
            return self.text_extractor.extract_text_from_pdf(file_path)
        elif extension == "docx":
            return self.extract_text_from_docx_mammoth(file_path)
        elif extension in ["png", "jpg"]:
            return self.text_extractor.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def extract_text_from_docx_mammoth(self, docx_path: str) -> str:
        """Extract text from a DOCX file using Mammoth."""
        with open(docx_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value

    def process_file_parallel(self, file_name: str, folder_path: str, pseudo_questions: list) -> list:
        """Process files in parallel."""
        try:
            file_path = os.path.join(folder_path, file_name)
            student_id, student_name = self.extract_name_and_id(file_name)
            answers = self.text_extractor.extract_student_answers(pseudo_questions, self.process_file(file_path))
            answers_list = answers.split("###")
            return [student_id, student_name] + [answer.strip() for answer in answers_list]
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None

    def read_questions(self, file_path: str) -> dict:
        """Read questions from a JSON file."""
        with open(file_path, 'r') as file:
            return json.load(file)

    def read_files_from_folder_parallel(self, folder_path: str, output_csv: str, pseudo_questions: list, max_workers=5):
        """Process multiple files in parallel and write results to CSV."""
        files = [file_name for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
        with open(output_csv, mode='a', newline='') as csv_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
            writer = csv.writer(csv_file)
            future_to_file = {executor.submit(self.process_file_parallel, file_name, folder_path, pseudo_questions): file_name for file_name in files}
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        writer.writerow(result)
                except Exception as exc:
                    print(f"Error processing file {file_name}: {exc}")

# Example usage:
def main():
    questions_path = '/path/to/questions.json'
    folder_path = "/path/to/files"
    output_csv = "all_students.csv"
    pseudo_questions = FileProcessor(TextExtractionService(api_key)).read_questions(questions_path)['pseudo_num']

    file_processor = FileProcessor(TextExtractionService(api_key))
    file_processor.read_files_from_folder_parallel(folder_path, output_csv, pseudo_questions)

if __name__ == "__main__":
    main()


class CacheEntry:
    def __init__(self, description: str):
        self.description = description
        self.timestamp = time.time()


class GeminiImageProcessor:
    CACHE_EXPIRATION = 30 * 60  # 30 minutes in seconds

    class Config(BaseModel):
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", "AIzaSyAOOkMaWs6tckmxnWvnFnmX-v08grr52IE"),
            description="Your Google API key for image processing",
        )

    def __init__(self):
        self.config = self.Config()
        self.image_cache = {}
        genai.configure(api_key=self.config.GOOGLE_API_KEY)

    def clean_expired_cache(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.image_cache.items()
            if current_time - entry.timestamp > self.CACHE_EXPIRATION
        ]
        for key in expired_keys:
            del self.image_cache[key]

    async def process_image(self, image_path: str) -> str:
        """Process a local image with Gemini and return its description."""
        try:
            if not self.config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for image processing")

            self.clean_expired_cache()

            if image_path in self.image_cache:
                print(f"Using cached description for {image_path[:30]}...")
                return self.image_cache[image_path].description

            # Check if the file exists
            if not os.path.exists(image_path):
                raise ValueError(f"File not found: {image_path}")

            # Open image and convert to bytes
            with Image.open(image_path) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                image_bytes = img_byte_arr.getvalue()

            # Create image content part
            image_part = {
                "mime_type": "image/jpeg",  # Change based on image type
                "data": image_bytes
            }

            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = "Give all the text present in this image without any extra parts."
            response = model.generate_content([prompt, image_part])

            description = response.text
            self.image_cache[image_path] = CacheEntry(description)

            # Remove oldest cache entry if cache exceeds 100 items
            if len(self.image_cache) > 100:
                oldest_key = min(self.image_cache, key=lambda k: self.image_cache[k].timestamp)
                del self.image_cache[oldest_key]

            return description
        except Exception as e:
            print(f"Error processing image with Gemini: {str(e)}")
            return f"[Error processing image: {str(e)}]"




async def main():
    processor = GeminiImageProcessor()
    image_url = "/Users/samhithdara/PycharmProjects/deepseek/docs/jamesaiden_200341_22284219_IMG_20230913_224153691.jpg"  # Replace with a real image URL
    description = await processor.process_image(image_url)
    print("Image Description:", description)

asyncio.run(main())



# class DeepSeek:
#     def __init__(self):
#         self.OLLAMA_URL = "http://localhost:11434/api/generate"
#         self.model = "deepseek-r1"
#
#     def query_ollama(self, prompt, stream=False):
#         # Please install OpenAI SDK first: `pip3 install openai`
#
#         # openai.api_key = "sk-bfb12a190e91406e9173755359593acc"
#         #
#         # response = openai.ChatCompletion.create(
#         #     model="deepseek-v3",
#         #     messages=[{"role": "user", "content": prompt}],
#         #     stream=stream
#         # )
#
#         # client = openai.OpenAI(api_key="sk-bfb12a190e91406e9173755359593acc", base_url="https://api.deepseek.com")
#         #
#         # response = client.chat.completions.create(
#         #     model="deepseek-v3",
#         #     messages=[
#         #         {"role": "system", "content": "You are a helpful assistant"},
#         #         {"role": "user", "content": prompt},
#         #     ],
#         #     stream=False
#         # )
#
#         # print(response.choices[0].message.content)
#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "stream": stream
#         }
#
#         response = requests.post(self.OLLAMA_URL, json=payload)
#
#         if response.status_code == 200:
#             return response.json().get("response", "No response from model.")
#         else:
#             return f"Error {response.status_code}: {response.text}"
#
#     def extract_student_answers(self, questions, answer):
#         num_questions = 12
#         query = f"""
#         Below is a student's answer PDF for {num_questions} questions. Extract the student's **exact text** for each question. Follow these rules:
#         1. Map answers strictly to the question numbers below.
#         2. If no answer exists for a question, leave it empty.
#         3. Do not modify the student's text (typos, symbols, or formatting must stay as-is).
#
#         ### Questions:
#         {questions}
#
#         ### Student's Answer PDF:
#         {answer}
#
#         ### Output Format:
#         1. [Exact text for Q1 or empty]
#         ###
#         2. [Exact text for Q2 or empty]
#         ###
#         ...
#         {num_questions}. [Exact text for QN or empty]
#         """
#         # query = (f"This is answer written by student {answer} for questions {questions} , "
#         #          f"Now can you extract what student has written for question considering the question, as it is without your alterations and if student didnt wrote for that question give empty ,"
#         #          f"each answer separated by `###`. Format like this and don't give your think section:\n"
#         #          f"1. [Answer to question 1]\n###\n2. [Answer to question 2]\n### ...and so on.")
#         answers = self.query_ollama(query)
#         answers_list = answers.split("###")
#         print(answers)
#         return answers_list


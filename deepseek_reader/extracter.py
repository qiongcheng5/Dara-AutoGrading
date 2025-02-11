import os
import time
import base64
import json
import csv
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pydantic import BaseModel, Field
import mammoth
import requests
# from openai import OpenAI  # openai version 1.1.1
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
import google.generativeai as genai  # assuming the Gemini API is imported here
import io


class CacheEntry:
    def __init__(self, description: str):
        self.description = description
        self.timestamp = time.time()


class GeminiImageProcessor:
    # CACHE_EXPIRATION = 30 * 60  # 30 minutes in seconds

    class Config(BaseModel):
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", "AIzaSyAOOkMaWs6tckmxnWvnFnmX-v08grr52IE"),
            description="Your Google API key for image processing",
        )

    def __init__(self):
        self.config = self.Config()
        # self.image_cache = {}
        genai.configure(api_key=self.config.GOOGLE_API_KEY)

    def process_image(self, image: Image.Image) -> str:
        """Process a provided image with Gemini and return its description."""
        try:
            if not self.config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for image processing")

            if image.format != 'JPEG':
                image = image.convert('RGB')  # Convert to RGB if it's not already

            # Convert image to bytes
            with io.BytesIO() as img_byte_arr:
                image.save(img_byte_arr, format="JPEG")  # Ensure it's saved as JPEG
                image_bytes = img_byte_arr.getvalue()

            # Create image content part
            image_part = {
                "mime_type": "image/jpeg",  # JPEG format
                "data": image_bytes
            }

            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = "Give all the text present in this image without any extra parts."
            response = model.generate_content([prompt, image_part])

            description = response.text
            return description
        except Exception as e:
            print(f"Error processing image with Gemini: {str(e)}")
            return f"[Error processing image: {str(e)}]"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        images = convert_from_path(pdf_path)
        pdf_text = ""
        for image in images:
            pdf_text += self.process_image(image)
        return pdf_text


class DeepSeek:
    def __init__(self):
        self.OLLAMA_URL = "http://localhost:11434/api/generate"
        self.model = "deepseek-r1"

    def query_ollama(self, prompt, stream=False):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }

        response = requests.post(self.OLLAMA_URL, json=payload)

        if response.status_code == 200:
            return response.json().get("response", "No response from model.")
        else:
            return f"Error {response.status_code}: {response.text}"

    def extract_student_answers(self, questions, answer):
        query = (f"This is answer pdf uploaded by student {answer} for questions {questions} , "
                 f"Now can you give me what student has written for each question as it is including what student tried ,"
                 f"basically segregate the given student text based on questions, separated by `###`. Format like this:\n"
                 f"1. [Answer to question 1]\n###\n2. [Answer to question 2]\n### ...and so on.")
        answers = self.query_ollama(query)
        answers_list = answers.split("###")
        print(answers)
        return answers_list


class FileProcessor:
    def __init__(self, gemini_processor: GeminiImageProcessor, deepseek: DeepSeek):
        # self.openai_processor = OpenAIProcessor(openai_api_key)
        self.gemini_processor = gemini_processor
        self.deepseek = deepseek

    def process_file(self, file_path: os.path) -> str:
        extension = file_path.split(".")[-1].lower()

        if extension == "html":
            return self.extract_text_from_html(file_path)
        elif extension == "pdf":
            return self.gemini_processor.extract_text_from_pdf(file_path)
        elif extension == "docx":
            return self.extract_text_from_docx_mammoth(file_path)
        elif extension == "png" or extension == "jpg":
            image = Image.open(file_path)
            return self.gemini_processor.process_image(image)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def extract_text_from_html(self, html_path: str) -> str:
        with open(html_path, "r", encoding="utf-8") as file:
            html_content = file.read()
            return self.convert_to_latex(html_content)

    def convert_to_latex(self, text: str) -> str:
        """Convert HTML-based math tags to LaTeX equivalents."""
        soup = BeautifulSoup(text, "html.parser")

        # Convert <sup> tags to LaTeX format for exponents
        for sup in soup.find_all("sup"):
            latex_sup = f"^{{{sup.get_text()}}}"
            sup.replace_with(latex_sup)

        # Convert <sub> tags to LaTeX format for subscripts
        for sub in soup.find_all("sub"):
            latex_sub = f"_{{{sub.get_text()}}}"
            sub.replace_with(latex_sub)

        # Convert <span class="frac"> to LaTeX format for fractions
        for frac in soup.find_all("span", class_="frac"):
            numerator = frac.find("span", class_="num")
            denominator = frac.find("span", class_="den")
            if numerator and denominator:
                latex_frac = r"\frac{" + numerator.get_text() + "}{" + denominator.get_text() + "}"
                frac.replace_with(latex_frac)

        # Convert <span class="sqrt"> to LaTeX format for square roots
        for sqrt in soup.find_all("span", class_="sqrt"):
            latex_sqrt = r"\sqrt{" + sqrt.get_text() + "}"
            sqrt.replace_with(latex_sqrt)

        return soup.get_text()

    def extract_text_from_docx_mammoth(self, docx_path: str) -> str:
        with open(docx_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value

    def extract_name_and_id(self, file_name):
        name_id = os.path.splitext(file_name)[0]
        splits = name_id.split('_')
        name, student_id = splits[0], splits[1]
        return student_id, name

    def process_file_parallel(self, file_name, folder_path, api_key=None, pseudo_questions=None):
        try:
            file_path = os.path.join(folder_path, file_name)
            student_id, student_name = self.extract_name_and_id(file_name)
            answers = self.deepseek.extract_student_answers(pseudo_questions, self.process_file(file_path))
            # print(self.process_file(file_path))
            # return [student_id, student_name, self.process_file(file_path)]
            # answers_list = answers.split("###")
            return [student_id, student_name] + [answer.strip() for answer in answers]
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None

    def read_files_from_folder_parallel(self, folder_path, output_csv, api_key, pseudo_questions, max_workers=5):
        files = [file_name for file_name in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, file_name))]

        with open(output_csv, mode='a', newline='') as csv_file, ThreadPoolExecutor(
                max_workers=max_workers) as executor:
            writer = csv.writer(csv_file)
            # Uncomment below to write header in CSV
            writer.writerow(['Student ID', 'Student Name', 'Answers',
                             "1. Solve the following recurrence relation using substitution method(?): x(n) = x(n-1) "
                             "+ 5 for n > 1, with x(1) = 0.\\n",
                             "2. Solve the following recurrence relation using substitution method(?): x(n) = 3x(n-1) "
                             "for n >= 1, with x(1) = 4.\\n",
                             "3. Solve the following recurrence relation using substitution method(?): x(n) = x(n-1) "
                             "+ n for n > 0, with x(0) = 0.\\n",
                             "4. Solve the following recurrence relation using substitution method(?): x(n) = n * x("
                             "n/3) + n for n > 1, with x(1) = 1 (solve for n=2^k).\\n",
                             "5. Solve the following recurrence relation using substitution method(?): x(n) = n * x("
                             "n/3) + n for n > 1, with x(1) = 1 (solve for n=3^k).\\n",
                             "6. Finding asymptotic annotation of the recursive algorithm. \\nAlgorithm Q(n)\\nInput- "
                             "A positive integer n\\nif n = 1 return 1\\nelse return Q(n - 1) + 2 * n - 1\\n. Set up "
                             "a recurrence relation for the values of the function Q(n) and solve it to determine "
                             "what this algorithm computes.\\n",
                             "7. Finding asymptotic annotation of the recursive algorithm. \\nAlgorithm Q(n)\\nInput- "
                             "A positive integer n\\nif n = 1 return 1\\nelse return Q(n - 1) + 2 * n - 1\\n. Set up "
                             "a recurrence relation for the number of multiplications made by this algorithm and "
                             "solve it.\\n",
                             "8. Finding asymptotic annotation of the recursive algorithm. \\nAlgorithm Q(n)\\nInput- "
                             "A positive integer n\\nif n = 1 return 1\\nelse return Q(n - 1) + 2 * n - 1\\n. Set up "
                             "a recurrence relation for the number of additions/subtractions made by this algorithm "
                             "and solve it.\\n",
                             "9. Use master method to analyze time complexity: Show that the solution of T(N) = 4T("
                             "N/2) + N is T(N) = O(N^2).\\n",
                             "10. Use master method to analyze time complexity: Show that the solution of T(N) = 2T("
                             "N/2) + N is T(N) = O(N log N).\\n",
                             "11. Use master method to analyze time complexity: Show that the solution of T(N) = 2T("
                             "N/2) + N^2 is T(N) = O(N^2).\\n",
                             "12. Use master method to analyze time complexity: Show that the solution of T(N) = 2T("
                             "N/2) + log2N is T(N) = O(N).\\n"])
            future_to_file = {
                executor.submit(self.process_file_parallel, file_name, folder_path, api_key, pseudo_questions):
                    file_name for file_name in files
            }

            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        writer.writerow(result)
                except Exception as exc:
                    print(f"Error processing file {file_name}: {exc}")


def main():
    gemini_processor = GeminiImageProcessor()
    deepseek = DeepSeek()
    file_processor = FileProcessor(gemini_processor, deepseek)
    output_csv = "all_students.csv"
    questions_path = '/Users/samhithdara/PycharmProjects/pdf_reader/actual-questions.json'
    with open(questions_path, 'r') as file:
        data = json.load(file)
    pseudo_questions = data['pseudo_num']
    file_processor.read_files_from_folder_parallel(folder_path="/Users/samhithdara/PycharmProjects/deepseek/docs",
                                                   output_csv=output_csv, api_key="", pseudo_questions=pseudo_questions)


main()

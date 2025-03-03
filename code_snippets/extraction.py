# The below code is used for extracting the text from students answers and storing to csv file using google gemini
# working properly might have limit for api usage.

import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pydantic import BaseModel, Field
import mammoth
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
import google.generativeai as genai
import io


class GeminiImageProcessor:

    class Config(BaseModel):
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", "key"),
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


class FileProcessor:
    def __init__(self, gemini_processor: GeminiImageProcessor):
        self.gemini_processor = gemini_processor

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

    def process_file_parallel(self, file_name, folder_path):
        try:
            file_path = os.path.join(folder_path, file_name)
            extracted_text = self.process_file(file_path)
            return [file_name, extracted_text]
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
            writer.writerow(['File name', 'extracted text'])
            # writer.writerow(['Student ID', 'Student Name', 'Answers',
            #                  "1. Solve the following recurrence relation using substitution method(?): x(n) = x(n-1) "
            #                  "+ 5 for n > 1, with x(1) = 0.\\n",
            #                  "2. Solve the following recurrence relation using substitution method(?): x(n) = 3x(n-1) "
            #                  "for n >= 1, with x(1) = 4.\\n",
            #                  "3. Solve the following recurrence relation using substitution method(?): x(n) = x(n-1) "
            #                  "+ n for n > 0, with x(0) = 0.\\n",
            #                  "4. Solve the following recurrence relation using substitution method(?): x(n) = n * x("
            #                  "n/3) + n for n > 1, with x(1) = 1 (solve for n=2^k).\\n",
            #                  "5. Solve the following recurrence relation using substitution method(?): x(n) = n * x("
            #                  "n/3) + n for n > 1, with x(1) = 1 (solve for n=3^k).\\n",
            #                  "6. Finding asymptotic annotation of the recursive algorithm. \\nAlgorithm Q(n)\\nInput- "
            #                  "A positive integer n\\nif n = 1 return 1\\nelse return Q(n - 1) + 2 * n - 1\\n. Set up "
            #                  "a recurrence relation for the values of the function Q(n) and solve it to determine "
            #                  "what this algorithm computes.\\n",
            #                  "7. Finding asymptotic annotation of the recursive algorithm. \\nAlgorithm Q(n)\\nInput- "
            #                  "A positive integer n\\nif n = 1 return 1\\nelse return Q(n - 1) + 2 * n - 1\\n. Set up "
            #                  "a recurrence relation for the number of multiplications made by this algorithm and "
            #                  "solve it.\\n",
            #                  "8. Finding asymptotic annotation of the recursive algorithm. \\nAlgorithm Q(n)\\nInput- "
            #                  "A positive integer n\\nif n = 1 return 1\\nelse return Q(n - 1) + 2 * n - 1\\n. Set up "
            #                  "a recurrence relation for the number of additions/subtractions made by this algorithm "
            #                  "and solve it.\\n",
            #                  "9. Use master method to analyze time complexity: Show that the solution of T(N) = 4T("
            #                  "N/2) + N is T(N) = O(N^2).\\n",
            #                  "10. Use master method to analyze time complexity: Show that the solution of T(N) = 2T("
            #                  "N/2) + N is T(N) = O(N log N).\\n",
            #                  "11. Use master method to analyze time complexity: Show that the solution of T(N) = 2T("
            #                  "N/2) + N^2 is T(N) = O(N^2).\\n",
            #                  "12. Use master method to analyze time complexity: Show that the solution of T(N) = 2T("
            #                  "N/2) + log2N is T(N) = O(N).\\n"])
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
    file_processor = FileProcessor(gemini_processor)
    output_csv = "extracted_text.csv"

    file_processor.read_files_from_folder_parallel(folder_path="/Users/samhithdara/PycharmProjects/deepseek/docs",
                                                   output_csv=output_csv)

main()

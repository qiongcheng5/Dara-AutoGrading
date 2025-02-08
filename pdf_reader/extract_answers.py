import base64
import csv
import json
import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from openai import OpenAI  # openai version 1.1.1
import requests
from pdf2image import convert_from_path
import mammoth
os.environ[
    "OPENAI_API_KEY"] = "key"  # getpass("OpenAI API Key: ")

api_key = os.getenv('OPENAI_API_KEY')

def encode_image_to_base64(image):
    # Convert the PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ''
    for image_path in images:
        base64_image = encode_image_to_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given a image of handwritten student answers, give me all the text present in the image and avoid extra lines."
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

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4-turbo",
            max_tokens=300
        )
        text += completion.choices[0].message.content
    print(f"extracted text from {pdf_path} -> {text}")
    return text

def convert_to_latex(text):
    """Convert HTML-based math tags to LaTeX equivalents."""
    soup = BeautifulSoup(text, "html.parser")

    # Convert <sup> tags to LaTeX format for exponents
    for sup in soup.find_all("sup"):
        latex_sup = f"^{{{sup.get_text()}}}"
        sup.replace_with(latex_sup)  # Replace <sup> tag content with LaTeX syntax

    # Convert <sub> tags to LaTeX format for subscripts
    for sub in soup.find_all("sub"):
        latex_sub = f"_{{{sub.get_text()}}}"
        sub.replace_with(latex_sub)  # Replace <sub> tag content with LaTeX syntax

    # Convert <span class="frac"> to LaTeX format for fractions
    for frac in soup.find_all("span", class_="frac"):
        numerator = frac.find("span", class_="num")
        denominator = frac.find("span", class_="den")
        if numerator and denominator:
            latex_frac = r"\frac{" + numerator.get_text() + "}{" + denominator.get_text() + "}"
            frac.replace_with(latex_frac)  # Replace <span class="frac"> with LaTeX syntax

    # Convert <span class="sqrt"> to LaTeX format for square roots
    for sqrt in soup.find_all("span", class_="sqrt"):
        latex_sqrt = r"\sqrt{" + sqrt.get_text() + "}"
        sqrt.replace_with(latex_sqrt)  # Replace <span class="sqrt"> with LaTeX syntax

    return soup.get_text()

def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()
        latex_text = convert_to_latex(html_content)
        return latex_text

def extract_text_from_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given a image of handwritten student answers, give me all the text present in the image and avoid extra lines."
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

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4-turbo",
            max_tokens=300
        )
        print('response', completion.choices[0].message.content)
        return completion.choices[0].message.content

def process_file(file_path):
    extension = file_path.split(".")[-1].lower()

    if extension == "html":
        return extract_text_from_html(file_path)
    elif extension == "pdf":
        return extract_text_from_pdf(file_path)
    elif extension == "docx":
        return extract_text_from_docx_mammoth(file_path)
    elif extension == "png" or extension == "jpg":
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def extract_name_and_id(file_name):
    name_id = os.path.splitext(file_name)[0]
    splits = name_id.split('_')
    name, student_id = splits[0], splits[1]
    return student_id, name

def extract_text_from_docx_mammoth(docx_path):
    with open(docx_path, "rb") as docx_file:
        result = mammoth.extract_raw_text(docx_file)
        return result.value

def extract_student_answers(questions, answer):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Given the answers {answer} written by students for questions {questions} , "
                            f"give me what student has answered for each question without manipulating text and complete text till the start of next question, separated by `###`. Format like this:\n"
                            "1. [Answer to question 1]\n###\n2. [Answer to question 2]\n### ...and so on."

                },
            ]
        }
    ]

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4-turbo"
    )
    print('response', completion.choices[0].message.content)
    return completion.choices[0].message.content

def read_questions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def process_file_parallel(file_name, folder_path, api_key, pseudo_questions):
    try:
        file_path = os.path.join(folder_path, file_name)
        student_id, student_name = extract_name_and_id(file_name)
        answers = extract_student_answers(pseudo_questions, process_file(file_path))
        answers_list = answers.split("###")
        return [student_id, student_name] + [answer.strip() for answer in answers_list]
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def read_files_from_folder_parallel(folder_path, output_csv, api_key, pseudo_questions, max_workers=5):
    files = [file_name for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]

    with open(output_csv, mode='a', newline='') as csv_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
        writer = csv.writer(csv_file)
        # Uncomment below to write header in CSV
        # writer.writerow(['Student ID', 'Student Name', 'Answers'])

        future_to_file = {executor.submit(process_file_parallel, file_name, folder_path, api_key, pseudo_questions): file_name for file_name in files}

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                result = future.result()
                if result:
                    writer.writerow(result)
            except Exception as exc:
                print(f"Error processing file {file_name}: {exc}")

# Example usage:
import time
start_time = time.time()
print(f"Start Time: {time.ctime(start_time)}")
questions_path = '/Users/samhithdara/PycharmProjects/pdf_reader/actual-questions.json'
questions_data = read_questions(questions_path)
pseudo_questions = questions_data['pseudo_num']
folder_path = "/Users/samhithdara/PycharmProjects/pdf_reader/pdf_reader/pseudo"
output_csv = "all_students.csv"

read_files_from_folder_parallel(folder_path, output_csv, api_key, pseudo_questions)

# Record the end time
end_time = time.time()
print(f"End Time: {time.ctime(end_time)}")

# Calculate and display the elapsed time
elapsed_time = end_time - start_time
print(f"Total Execution Time: {elapsed_time:.2f} seconds")

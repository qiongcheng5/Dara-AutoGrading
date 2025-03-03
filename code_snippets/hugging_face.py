# After extracted text to segregate and then grade below i am trying to segregate the text from student answers using hugging face models.
# Not able to get the segregated results likely due to the fact that the question-answering (QA) model (deepset/roberta-base-squad2) is designed to extract answers from a context that contains the answer explicitly

import pandas as pd
from transformers import pipeline

# Load the free QA model from Hugging Face
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load extracted text from CSV
input_csv = "/Users/samhithdara/PycharmProjects/deepseek/all_students.csv"
df = pd.read_csv(input_csv)

questions = [
    "Solve the following recurrence relation using substitution method: x(n) = x(n-1) + 5 for n > 1, with x(1) = 0.",
    "Solve the following recurrence relation using substitution method: x(n) = 3x(n-1) for n >= 1, with x(1) = 4.",
    "Solve the following recurrence relation using substitution method: x(n) = x(n-1) + n for n > 0, with x(0) = 0.",
]

# Function to extract answers using NLP model
def extract_answers(text, question_list):
    extracted = {}
    for q in question_list:
        response = qa_model(question=q, context=text)
        extracted[q] = response["answer"] if response["score"] > 0.5 else "Not found"
    return extracted

# Process each row and extract answers
output_data = []
for _, row in df.iterrows():
    file_name, extracted_text = row["File name"], row["extracted text"]
    answers = extract_answers(extracted_text, questions)
    output_data.append([file_name] + [answers[q] for q in questions])

# Save results to a new CSV
output_df = pd.DataFrame(output_data, columns=["File Name"] + questions)
output_df.to_csv("segregated_answers.csv", index=False)
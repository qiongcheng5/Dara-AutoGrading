import json
import os

import google.generativeai as genai
from langchain.vectorstores import Chroma
import re
import time
import math
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from dotenv import load_dotenv
from ragas import SingleTurnSample

load_dotenv()
# Configure Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Replace with your actual API key

# Initialize ChromaDB
persist_directory = "/Users/samhithdara/PycharmProjects/llama/docs/chroma/"
vectordb = Chroma(persist_directory=persist_directory)
test_data_samples = []

#  Retrieve context from ChromaDB
def vector_query(query):
    """Fetch relevant text from ChromaDB for the given query."""
    response = genai.embed_content(
        model="models/embedding-001",  # Use Gemini embedding model
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = response["embedding"]
    docs_with_scores = vectordb.similarity_search_by_vector(query_embedding, k=10)  # Fetch top 10
    # context = "\n".join([doc.page_content for doc, score in docs_with_scores if score > 0.7])
    context = "\n".join([doc.page_content for doc in docs_with_scores])  # No score unpacking

    return context if context else "No relevant context found."


#  Grade answer using Gemini API
def grade_answer(question, answer, SLO,file_name, add_new=False):
    """Uses Google Gemini to evaluate a student's response based on the given question and rubric."""

    print("Grading:", question)

    # Create grading prompt
    query = (
        f"In a graduate Algorithms and Data Structures course, students are asked the following question: {question}\n"
        f"Please review the student response and assess their ability to answer the asymptotic analysis.\n"
        f"Grading Rubric:\n {SLO}\n"
        f"Student Response:\n {answer}\n"
        "Answer in format: <rating>\n<explanation>.\n"
        "The first line should ONLY have a number for rating. The second line should have its corresponding explanation."
    )

    # Retrieve reference material from ChromaDB
    retrieved_context = vector_query(query)
    context_message = f"Here is the provided context: {retrieved_context}"

    # Send the query to Google Gemini for grading
    model = genai.GenerativeModel("gemini-1.5-pro")  # Latest model
    max_retries = 5
    retry_delay = 10  # Start with 10 seconds delay
    for attempt in range(max_retries):
        try:
            response = model.generate_content([query,context_message])
            result = response.text.strip() if response.text else "0\nNo response generated."

            # Extract score and explanation
            match = re.search(r'\b\d+\b', result)
            score = float(match.group()) if match else 0
            explanation = result.split("\n", 1)[1] if "\n" in result else "No explanation provided."

            print(score)
            if add_new:
                test_data_samples.append([file_name, SingleTurnSample( user_input=query, response=result, reference=context_message)])
            return score, explanation

        except Exception as e:  #  Catch all exceptions
            error_type = type(e).__name__  #  Get exception type dynamically
            error_message = str(e)
            print(f"Exception Type: {error_type} | Message: {error_message}")

            if "429" in error_message or "quota" in error_message.lower():

                print(
                    f"Rate limit exceeded! Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")

                time.sleep(retry_delay)

                retry_delay *= 2  # Exponential backoff

            else:

                print("Unexpected error, not retrying.")

                return 0, f"Error: {error_message}"
    response = model.generate_content(
        [query, context_message]
    )

    result = response.text.strip() if response.text else "0\nNo response generated."

    # Extract score and explanation
    match = re.search(r'\b\d+\b', result)
    score = float(match.group()) if match else 0  # Extract first number as score
    explanation = result.split("\n", 1)[1] if "\n" in result else "No explanation provided."
    print(score)
    return score, explanation

def load_questions():
    json_file_path = '/Users/samhithdara/PycharmProjects/pdf_reader/questions.json'
    with open(json_file_path, 'r') as json_file:
        questions = json.load(json_file)
    return questions


def load_slo():
    json_file_path = '/Users/samhithdara/PycharmProjects/pdf_reader/Rubrics.json'
    with open(json_file_path, 'r') as json_file:
        slo = json.loads(json_file.read())
    return slo

#  Process grading for each student
def process_grading(q_idx, q, slo_name, slo, data):
    grade_df = []
    grade_details_list = {}
    slo_final_scores = {}

    for index, row in data.iterrows():
        col = q["question_col"]
        if col not in data.columns:
            continue

        scores, explanations = [], []
        for i in range(3):  # Run grading 3 times
            # try:
            time.sleep(3)
            if i==1:
                score, explanation = grade_answer(col, row[col], slo[slo_name],row["File name"], True)
            else:
                score, explanation = grade_answer(col, row[col], slo[slo_name],row["File name"])
            # except Exception as e:
            #     print(e)
            #     score, explanation = 0, "N/A"

            scores.append(score)
            explanations.append(explanation)

        # Median of 3 scores (removing extreme values)
        sorted_scores = sorted(scores)[1:-1] if len(scores) >= 3 else scores
        final_score = sorted_scores[0] if sorted_scores else 0
        expln_idx = scores.index(final_score)

        # Store grading details
        grade_df.append([
            row["File name"], row[col], *scores, *explanations
        ])
        grade_details_list[row["File name"]] = [final_score, explanations[expln_idx]]

        # Store final SLO score
        slo_final_scores.setdefault(row["File name"], {}).setdefault(slo_name, 0)
        slo_final_scores[row["File name"]][slo_name] += final_score / q["max_score"]

    # Create DataFrames
    score_df = pd.DataFrame(grade_df, columns=[
        "File name", f"{slo_name}_Answer",
        f"{slo_name}_Score1", f"{slo_name}_Score2", f"{slo_name}_Score3",
        f"{slo_name}_Explanation1", f"{slo_name}_Explanation2", f"{slo_name}_Explanation3"
    ])
    grade_details_df = pd.DataFrame(
        [(name, *details) for name, details in grade_details_list.items()],
        columns=["File name", f"{slo_name}_Final_Score", f"{slo_name}_Final_Reasoning"]
    )

    return score_df, grade_details_df, slo_final_scores


#  Main function to handle grading for all students
def Grade():
    start_time = time.time()
    print(f"Start Time: {time.ctime(start_time)}")

    # Load questions and rubrics
    questions, slo = load_questions(), load_slo()

    # Load student responses
    file_path = "segregated.csv"
    if not os.path.isfile(file_path):
        print("File not found.")
        return
    data = pd.read_csv(file_path, delimiter=',', encoding='ISO-8859-1')

    all_dfs, all_dfs_gradedetails = [], []
    slo_final_scores = {}

    for q_idx, q in enumerate(questions):
        slo_name = q['Rubric_name']
        if q["question_col"] not in data.columns:
            print(f"Column {q['question_col']} not found in data")
            continue
        score_df, grade_details_df, partial_slo_scores = process_grading(q_idx, q, slo_name, slo, data)
        all_dfs.append(score_df)
        all_dfs_gradedetails.append(grade_details_df)
        for student, scores in partial_slo_scores.items():
            slo_final_scores.setdefault(student, {}).update(scores)

    # Save grading results to Excel
    fixed_data = data[["File name"]].set_index("File name")
    slo_df = pd.DataFrame.from_dict(slo_final_scores, orient='index').reset_index().rename(
        columns={'index': 'File name'})

    df = reduce(lambda x, y: x.merge(y, on='File name'), [fixed_data] + all_dfs)
    gd_df = reduce(lambda x, y: x.merge(y, on='File name'), [fixed_data] + all_dfs_gradedetails)
    slo_final = reduce(lambda x, y: x.merge(y, on='File name'), [fixed_data, slo_df])

    output_path = 'graded.xlsx'
    with pd.ExcelWriter(output_path) as excel_writer:
        df.to_excel(excel_writer, sheet_name='grading_details', index=False)
        gd_df.to_excel(excel_writer, sheet_name='SLOs_scores_details', index=False)
        slo_final.to_excel(excel_writer, sheet_name='SLOs_scores', index=False)
    csv_file="reference_data_temp.csv"
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file, escapechar="\\", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["File Name", "User Input", "Response", "Reference"])  # Header row
        for sample in test_data_samples:
            writer.writerow([sample[0], sample[1].user_input, sample[1].response, sample[1].reference])

    print(f"Processing complete. Output saved to {output_path}")
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")


#  Run grading process
if __name__ == "__main__":
    Grade()
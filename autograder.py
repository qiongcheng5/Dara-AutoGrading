import pandas as pd
import math
import os
import numpy as np
import json
import re
from functools import reduce

from openai import OpenAI
from retry import retry
import openai


# Load JSON data
def load_questions():
    json_file_path = 'questions.json'
    with open(json_file_path, 'r') as json_file:
        questions = json.load(json_file)
    return questions


def load_slo():
    json_file_path = 'Rubrics.json'
    with open(json_file_path, 'r') as json_file:
        slo = json.loads(json_file.read())
    return slo


# Grade answers
@retry(tries=3, delay=3)
def grade_answer(question, answer, SLO):
    context = "In a graduate Algorithms and Data Structures course, students are asked the following question: {} \n "\
            "Please review the student response and assess their ability to answer the asymptotic analysis\n" \
             "Grading Rubric:\n {}".format(question, SLO)

    # context = "In a graduate Algorithms and Data Structures course, students are asked the following question: {} \n "\
    #         "Please review the student response and assess their ability to compare the asymptotic growth rates of these" \
    #         " two functions. \n" \
    #         "Rate the student’s response on a scale from 0 to 3 and provide your reasoning. \n " \
    #          "Grading Rubric:\n {}".format(question, SLO)


    answer_msg = "Student Response:\n {}" \
                 "Answer in format: <rating>\n<explanation>. " \
                 "The first line should ONLY have a number for rating. The second line should have its corresponding explanation.".format(
        answer)
    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": answer_msg},
    ]

    client = OpenAI(
        # This is the default and can be omitted
        api_key="api_key"
    )
    completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4-turbo",
    )

    result = completion.choices[0].message.content
    score, explanation = result.split("\n", 1)
    # print("score - ",score, "Explaination - ", explanation)
    pattern = r'\b\d+\b'
    match = re.search(pattern, score)
    if match:
        score = float(match.group())

    return float(score), explanation


def main():
    questions, slo = load_questions(), load_slo()

    file_path = "pdf_reader/pseudo_parallel.csv"
    if not os.path.isfile(file_path):
        print("File not found.")
        return

    # Read the CSV file using pandas
    data = pd.read_csv(file_path,delimiter=',', encoding='ISO-8859-1')

    # def extract_question(column_name):
    #     return column_name.split(': ', 1)[-1]

    # data = data.rename(columns=lambda x: extract_question(x) if ': ' in x else x)

    all_dfs = []
    all_dfs_gradedetails = []
    slo_counts = {}
    slo_final_scores = {}

    for q_idx, q in enumerate(questions):
        col = q["question_col"]
        if col not in data.columns:
            # print(col)
            continue
        grade_df = []
        grade_details_list = []
        # slo_name = f"SLO{q_idx}"
        slo_name = q['Rubric_name']
        # canvas_question_id = col[:9]
        slo_counts[slo_name] = slo_counts.get(slo_name, 0) + 1
        for index, row in data.iterrows():
            lowest_score = math.inf
            lowest_explanation = None
            scores = []
            explanations = []
            for i in range(3):
                grade = None
                try:
                    grade = grade_answer(col, row[col], slo[slo_name])
                    if grade and len(grade) != 2:
                        print("Error in GPT response: Response format")
                        score, explanation = 0, "N/A"
                    else:
                        score, explanation = grade
                except Exception as e:
                    print("Error in GPT response: Exception", e)
                    score, explanation = 0, "N/A"
                print("Row:", index, "\tIter:", i, "\tQ:", q_idx, "\tScore:", score)
                score = float(score)
                scores.append(score)
                explanations.append(explanation)
                if score < lowest_score:
                    lowest_score = score
                    lowest_explanation = explanation
            sorted_scores = scores
            if len(scores) >= 3:
                sorted_scores = sorted(scores)[1:-1]
            expln_idx = scores.index(sorted_scores[0])
            grade_df.append([
                row["Student Name"],
                row[col],
                scores[0],
                explanations[0],
                scores[1],
                explanations[1],
                scores[2],
                explanations[2]
            ])
            grade_details_list.append([
                row["Student Name"],
                sorted_scores[0],
                explanations[expln_idx]
            ])

            if row["Student Name"] not in slo_final_scores:
                slo_final_scores[row["Student Name"]] = {}
            slo_final_scores[row["Student Name"]][slo_name] = slo_final_scores[row["Student Name"]].get(slo_name, 0) + (
                        sorted_scores[0] / q["max_score"])

        score_df = pd.DataFrame(
            grade_df,
            columns=[
                "Student Name",
                f"{slo_name}_{slo_counts[slo_name]}_Answer",
                f"{slo_name}_{slo_counts[slo_name]}_Score1_({q['max_score']})",
                f"{slo_name}_{slo_counts[slo_name]}_Score1_Reasoning",
                f"{slo_name}_{slo_counts[slo_name]}_Score2_({q['max_score']})",
                f"{slo_name}_{slo_counts[slo_name]}_Score2_Reasoning",
                f"{slo_name}_{slo_counts[slo_name]}_Score3_({q['max_score']})",
                f"{slo_name}_{slo_counts[slo_name]}_Score3_Reasoning",
            ],
        )
        grade_details_df = pd.DataFrame(
            grade_details_list,
            columns=[
                "Student Name",
                f"{slo_name}_{slo_counts[slo_name]}_Score_({q['max_score']})",
                f"{slo_name}_{slo_counts[slo_name]}_Score_Reasoning",
            ],
        )
        all_dfs.append(score_df)
        all_dfs_gradedetails.append(grade_details_df)

    # FIXED_FIELDS = ["Student Name", "id", "sis_id", "section", "section_id", "section_sis_id", "submitted", "attempt"]
    FIXED_FIELDS = ["Student Name"]

    fixed_data = data[FIXED_FIELDS]
    fixed_data.set_index("Student Name")


    slo_df = pd.DataFrame.from_dict(slo_final_scores, orient='index').reset_index()
    slo_df = slo_df.rename(columns={'index': 'Student Name'})

    df = reduce(lambda x, y: x.merge(y, on='Student Name'), [fixed_data] + all_dfs)
    gd_df = reduce(lambda x, y: x.merge(y, on='Student Name'), [fixed_data] + all_dfs_gradedetails)
    slo_final = reduce(lambda x, y: x.merge(y, on='Student Name'), [fixed_data, slo_df])

    def categorize_value(value):
        if value >= 0.9:
            return 10
        elif value >= 0.6:
            return 8
        elif value >= 0.35:
            return 6
        else:
            return 4

    slo_final.loc[:, slo_final.columns.str.startswith('SLO')] = slo_final.loc[:,
                                                                slo_final.columns.str.startswith('SLO')].applymap(
        categorize_value)

    output_path = 'pdf_reader/output.xlsx'
    with pd.ExcelWriter(output_path) as excel_writer:
        df.to_excel(excel_writer, sheet_name='grading_details', index=False)
        gd_df.to_excel(excel_writer, sheet_name='SLOs_scores_details', index=False)
        slo_final.to_excel(excel_writer, sheet_name='SLOs_scores', index=False)

    print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    main()
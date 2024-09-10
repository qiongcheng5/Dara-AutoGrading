from flask import Blueprint, request, jsonify, render_template, redirect, session, Response, send_file
from models.models import AutogradeInput
from extensions import db
import openai
import pandas as pd
from config import Config
import math
import os
import numpy as np
import json
import zipfile
import re
from functools import reduce
from retry import retry


autograde_api = Blueprint("autograde", __name__)

GRADING_TIMES = Config.GRADING_TIMES
FIXED_FIELDS = ["name", "id", "sis_id", "section", "section_id", "section_sis_id", "submitted", "attempt"]

def load_questions():
    # Specify the path to your JSON file
    json_file_path = 'static/questions.json'

    # Open the JSON file and load its content into a dictionary
    with open(json_file_path, 'r') as json_file:
        questions = json.load(json_file)

    return questions

def load_slo():
    # Specify the path to your JSON file
    json_file_path = 'static/SLO.json'

    # Open the JSON file and load its content into a dictionary
    with open(json_file_path, 'r') as json_file:
        slo = json.loads(json_file.read())

    return slo

def load_closed_questions():
    # Specify the path to your JSON file
    json_file_path = 'static/closed-questions.json'

    # Open the JSON file and load its content into a dictionary
    with open(json_file_path, 'r') as json_file:
        questions = json.load(json_file)

    return questions

@autograde_api.route("/autograde", methods=["GET", "POST"])
def instructor():
    questions, slo = load_questions(), load_slo()
    if request.method == "POST":
        file = request.files["studentfile"]
        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read the CSV file using pandas
        data = pd.read_csv(file_path, encoding = 'unicode_escape')

        # Remove question ids from columns to generalize input files for all sections
        # Define a function to extract the part after the colon
        def extract_question(column_name):
            return column_name.split(': ', 1)[-1]
        # Use the rename function with the lambda function
        data = data.rename(columns=lambda x: extract_question(x) if ': ' in x else x)

        # df for all grading scores
        all_dfs = []
        # df for average slo scores
        all_dfs_gradedetails = []
        slo_counts = {}
        # df for 1-4 scores
        slo_final_scores = {
            # will have format
            # <student name> : { # dict of slo names mapping to list of scores for this student }
        }

        for q_idx, q in enumerate(questions):
            col = q["question_col"]
            if col not in data.columns:
                continue
            grade_df = []
            grade_details_list = []
            slo_name = request.form[f"slo-{q_idx}"]
            canvas_question_id = col[:9]
            slo_counts[slo_name] = slo_counts.get(slo_name, 0) + 1
            for index, row in data.iterrows():
                lowest_score = math.inf
                lowest_explanation = None
                scores = []
                explanations = []
                for i in range(GRADING_TIMES):
                    grade = None
                    try:
                        grade = grade_answer(col, row[col], slo[slo_name])
                        if grade and len(grade) != 2:
                            print("Error in GPT response: Response format")
                            score, explanation = 0, "N/A"
                        else:
                            score, explanation = grade
                    except Exception:
                        print("Error in GPT response: Exception")
                        score, explanation = 0, "N/A"
                    print("Row:", index, "\tIter:", i, "\tQ:", q_idx, "\tScore:", score)
                    score = float(score)
                    scores.append(score)
                    explanations.append(explanation)
                    if score < lowest_score:
                        lowest_score = score
                        lowest_explanation = explanation
                # remove high and low values
                sorted_scores = scores
                if len(scores) >= 3:
                    sorted_scores = sorted(scores)[1:-1]
                expln_idx = scores.index(sorted_scores[0])
                grade_df.append([
                    row["name"], 
                    row[col], 
                    scores[0],
                    explanations[0],
                    scores[1],
                    explanations[1],
                    scores[2],
                    explanations[2]
                ])
                grade_details_list.append([
                    row["name"], 
                    sorted_scores[0],
                    explanations[expln_idx]
                ])

                # construct df for 1-4 slo scores
                if row["name"] not in slo_final_scores:
                    slo_final_scores[row["name"]] = {}
                slo_final_scores[row["name"]][slo_name] = slo_final_scores[row["name"]].get(slo_name, 0) + (sorted_scores[0] / q["max_score"])

            score_df = pd.DataFrame(
                grade_df, 
                columns=[
                    "name", 
                    f"{slo_name}_{slo_counts[slo_name]}_Answer", 
                    f"{slo_name}_{slo_counts[slo_name]}_Score1_({q['max_score']})",
                    f"{slo_name}_{slo_counts[slo_name]}_Score1_Reasoning", 
                    f"{slo_name}_{slo_counts[slo_name]}_Score2_({q['max_score']})",
                    f"{slo_name}_{slo_counts[slo_name]}_Score2_Reasoning", 
                    f"{slo_name}_{slo_counts[slo_name]}_Score3_({q['max_score']})",
                    f"{slo_name}_{slo_counts[slo_name]}_Score3_Reasoning", 
                ], 
            )
            score_df.set_index("name")
            # score_df[f"{canvas_question_id}_question: {slo_name}"] = col
            # score_df.insert(1, f"{canvas_question_id}_question: {slo_name}", score_df.pop(f"{canvas_question_id}_question: {slo_name}"))
            grade_details_df = pd.DataFrame(
                grade_details_list,
                columns=[
                    "name", 
                    f"{slo_name}_{slo_counts[slo_name]}_Score_({q['max_score']})",
                    f"{slo_name}_{slo_counts[slo_name]}_Score_Reasoning", 
                ],
            )
            grade_details_df.set_index("name")
            all_dfs.append(score_df)
            all_dfs_gradedetails.append(grade_details_df)

        # fixed columns name, id, sis, etc.
        fixed_data = data[FIXED_FIELDS]
        fixed_data.set_index("name")

        # close ended columns
        # fetches the pre-graded canvas scores
        close_ended = load_closed_questions()
        close_ended_mapper = {}
        for q in close_ended:
            close_ended_mapper[q["answer_col"]] = q["output_col"]

        closed_ended_cols = ["name"] + list(map(lambda x : x["answer_col"], close_ended))
        close_ended_data_df = data[closed_ended_cols]
        close_ended_data_df.set_index("name")
        close_ended_data_df = close_ended_data_df.rename(columns=close_ended_mapper)

        # 1-4 range slo scores
        slo_df = pd.DataFrame.from_dict(slo_final_scores, orient='index').reset_index()
        slo_df = slo_df.rename(columns={'index': 'name'})
        for col in slo_df.columns:
            if col == "name":
                continue
            slo_df[col] = (slo_df[col] / slo_counts[col]) * 4

        close_ended_colgroups = {}
        for key in close_ended:
            if key["SLO_name"] in close_ended_colgroups:
                close_ended_colgroups[key["SLO_name"]].append(key)
            else:
                close_ended_colgroups[key["SLO_name"]] = [key]

        temp_df = pd.DataFrame()
        for slo in close_ended_colgroups:
            group = close_ended_colgroups[slo]
            temp_df["name"] = close_ended_data_df["name"]
            temp_df[slo] = 0
            for q in group:
                temp_df[slo] += (close_ended_data_df[q["output_col"]] / q["max_score"])
            temp_df[slo] = (temp_df[slo] / len(group)) * 4
                

        # merge all dfs
        df = reduce(lambda x, y: x.merge(y, on='name'), [fixed_data, close_ended_data_df] + all_dfs)
        gd_df = reduce(lambda x, y: x.merge(y, on='name'), [fixed_data, close_ended_data_df] + all_dfs_gradedetails)
        slo_final = reduce(lambda x, y: x.merge(y, on='name'), [fixed_data, temp_df, slo_df])

        # Convert slo values to categorical
        # Define a function to categorize values based on the given thresholds
        def categorize_value(value):
            if value >= 4 * 0.9:
                return 4
            elif value >= 4 * 0.6:
                return 3
            elif value >= 4 * 0.35:
                return 2
            else:
                return 1

        # Apply the categorize_value function to columns starting with "SLO"
        slo_final.loc[:, slo_final.columns.str.startswith('SLO')] = slo_final.loc[:, slo_final.columns.str.startswith('SLO')].applymap(categorize_value)

        # Export DataFrames to Excel using ExcelWriter
        with pd.ExcelWriter('downloads/output.xlsx') as excel_writer:
            df.to_excel(excel_writer, sheet_name='grading_details', index=False)
            gd_df.to_excel(excel_writer, sheet_name='SLOs_scores_details', index=False)
            slo_final.to_excel(excel_writer, sheet_name='SLOs_scores', index=False)

        email = request.form["email"]
        canvas_crn = request.form["class"]
        session = request.form["session_id"]
        if session == "":
            session = None
        db_obj = AutogradeInput(email, canvas_crn, session, file_path)
        db.session.add(db_obj)
        db.session.commit()
        
        return send_file('downloads/output.xlsx',
                mimetype='xlsx',
                download_name='output.xlsx',
                as_attachment=True)
        
    return render_template("autograde.html", questions=questions, slo=slo)


@autograde_api.route("/download/<int:index>", methods=["GET", "POST"])
def download(index):
    df = session[f"df_{index}"]
    return Response(
        df,
        mimetype="text/csv",
        headers={"Content-disposition":
        "attachment; filename=grades.csv"}
    )

@retry(tries=3, delay=3)
def grade_answer(question, answer, SLO):
    context = "{}\n Read the student response, grade student skill in {}. \n" \
            "Rate student reponse between 0 through 10, and give your reason for the rating.\n".format(question, SLO)
    answer_msg = "Student Response:\n {}" \
            "Answer in format: <rating>\n<explanation>. " \
            "The first line should ONLY have a number for rating. The second line should have its corresponding explanation.".format(answer)
    messages = [
        { "role": "user", "content": context },
        { "role": "user", "content": answer_msg },
    ]

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )

    result = completion.choices[0].message.content
    score, explanation = result.split("\n", 1)
    pattern = r'\b\d+\b'
    match = re.search(pattern, score)
    if match:
        score = float(match.group())

    return float(score), explanation
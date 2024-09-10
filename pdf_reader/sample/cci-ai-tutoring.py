import os

import streamlit as st
from streamlit_chat import message
from pathlib import Path

import openai
from dotenv import load_dotenv
import requests

st.set_page_config(
    page_title="Reflection chatbot",
    page_icon=":robot:"
)

#load_dotenv(dotenv_path=dotenv_path)
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Setting the path to the env file and then loading up the API key from the file
dotenv_path = Path('/home/vagrant/chunting/prjbox/exp_ai/OpenAI/.env')
load_dotenv(dotenv_path=dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.header("AI Tutoring")
st.markdown("Write down your concern about a computer science course and ask a question. ")
st.markdown("For example, I have a problem with object-oriented programming in python. Please give an example.")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

#
def query(payload):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are ChatGPT, a large language model trained by OpenAI. "
                            "You will answer the users queries."},

                {"role": "user",
                 "content": payload["inputs"]["text"]},

                     # "Here is a list of IDs and associated learning outcome from data structures and algorithms topics:\n"
                     #        + learning_outcome +
                     #        "\n Given the below code, only list the ID numbers covered by the code in the format '1,2,3':" + codefile},
                # {"role": "assistant", "content": "ID-Learning Concept\n " + learning_concepts},
                # {"role": "user", "content": "Only give the IDs for the corres: " + codefile}
            ]
        )
        return response['choices'][0]['message']['content']

    except Exception as e:
        print(e)
        return str(e)

def get_text():
    input_text = st.text_input("", key="input") # value="I have a problem with object-oriented programming in python. Please give an example.")
    return input_text


user_input = get_text()

if user_input:
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        }
    })

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'] ) -1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

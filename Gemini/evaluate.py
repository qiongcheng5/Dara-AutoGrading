import pandas as pd
from sentence_transformers import SentenceTransformer, util
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load models
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # SBERT variant for cosine similarity
distilbert_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")  # DistilBERT for semantic similarity
minilm_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # MiniLM for semantic similarity
fasttext_model = fasttext.load_model('/Users/samhithdara/Downloads/cc.en.300.bin')  # Load FastText pre-trained model

# Load CSV file
input_file = "/Users/samhithdara/PycharmProjects/pdf_reader/Gemini/reference_Data.csv"  # Change this to your file path
df = pd.read_csv(input_file, nrows=16)  # Load the first 10 rows for testing


# Function to compute Cosine Similarity using SBERT
def compute_cosine_similarity(text1, text2, model):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()  # Cosine similarity score (0 to 1)


# Function to compute Cosine Similarity using FastText
# Function to compute Cosine Similarity using FastText
def compute_fasttext_similarity(text1, text2, model):
    # Remove any newline characters from the text
    text1 = text1.replace("\n", " ")
    text2 = text2.replace("\n", " ")

    # Get FastText embeddings for sentences
    emb1 = model.get_sentence_vector(text1)
    emb2 = model.get_sentence_vector(text2)

    # Compute cosine similarity
    return cosine_similarity([emb1], [emb2])[0][0]  # Cosine similarity score (0 to 1)



# Function to compute Cosine Similarity using GloVe
def compute_glove_similarity(text1, text2, glove_model):
    def get_sentence_vector(sentence, glove_model):
        words = sentence.split()
        vectors = [glove_model[word] for word in words if word in glove_model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(50)  # Default vector size 50

    emb1 = get_sentence_vector(text1, glove_model)
    emb2 = get_sentence_vector(text2, glove_model)
    return cosine_similarity([emb1], [emb2])[0][0]  # Cosine similarity score (0 to 1)


# Process each row and compute scores for all models
scores_cosine = []
scores_distilbert = []
scores_minilm = []
scores_fasttext = []

for _, row in df.iterrows():
    user_input = str(row["User Input"])
    reference = str(row["Reference"])
    response = str(row["Response"])
    ground_truth = str(row["Ground Truth"])

    # Compute SBERT Cosine Similarity Scores
    relevance_cosine = compute_cosine_similarity(user_input, reference, sbert_model)  # Retrieval relevance
    correctness_cosine = compute_cosine_similarity(response, ground_truth, sbert_model)  # Answer correctness
    context_recall_cosine = compute_cosine_similarity(reference, ground_truth, sbert_model)  # Context recall
    faithfulness_cosine = compute_cosine_similarity(response, reference, sbert_model)  # Faithfulness

    # Compute DistilBERT Cosine Similarity Scores
    relevance_distilbert = compute_cosine_similarity(user_input, reference, distilbert_model)
    correctness_distilbert = compute_cosine_similarity(response, ground_truth, distilbert_model)
    context_recall_distilbert = compute_cosine_similarity(reference, ground_truth, distilbert_model)
    faithfulness_distilbert = compute_cosine_similarity(response, reference, distilbert_model)

    # Compute MiniLM Cosine Similarity Scores
    relevance_minilm = compute_cosine_similarity(user_input, reference, minilm_model)
    correctness_minilm = compute_cosine_similarity(response, ground_truth, minilm_model)
    context_recall_minilm = compute_cosine_similarity(reference, ground_truth, minilm_model)
    faithfulness_minilm = compute_cosine_similarity(response, reference, minilm_model)

    # Compute FastText Cosine Similarity Scores
    relevance_fasttext = compute_fasttext_similarity(user_input, reference, fasttext_model)
    correctness_fasttext = compute_fasttext_similarity(response, ground_truth, fasttext_model)
    context_recall_fasttext = compute_fasttext_similarity(reference, ground_truth, fasttext_model)
    faithfulness_fasttext = compute_fasttext_similarity(response, reference, fasttext_model)

    # Append scores for each metric to respective lists
    scores_cosine.append([
        user_input,
        relevance_cosine, correctness_cosine, context_recall_cosine, faithfulness_cosine
    ])

    scores_distilbert.append([
        user_input,
        relevance_distilbert, correctness_distilbert, context_recall_distilbert, faithfulness_distilbert
    ])

    scores_minilm.append([
        user_input,
        relevance_minilm, correctness_minilm, context_recall_minilm, faithfulness_minilm
    ])

    scores_fasttext.append([
        user_input,
        relevance_fasttext, correctness_fasttext, context_recall_fasttext, faithfulness_fasttext
    ])

# Convert to DataFrames for each sheet
df_cosine = pd.DataFrame(scores_cosine, columns=[
    "User Input", "Relevance (SBERT)", "Correctness (SBERT)", "Context Recall (SBERT)", "Faithfulness (SBERT)"
])

df_distilbert = pd.DataFrame(scores_distilbert, columns=[
    "User Input", "Relevance (DistilBERT)", "Correctness (DistilBERT)", "Context Recall (DistilBERT)",
    "Faithfulness (DistilBERT)"
])

df_minilm = pd.DataFrame(scores_minilm, columns=[
    "User Input", "Relevance (MiniLM)", "Correctness (MiniLM)", "Context Recall (MiniLM)", "Faithfulness (MiniLM)"
])

df_fasttext = pd.DataFrame(scores_fasttext, columns=[
    "User Input", "Relevance (FastText)", "Correctness (FastText)", "Context Recall (FastText)",
    "Faithfulness (FastText)"
])

# Save to Excel with multiple sheets
with pd.ExcelWriter("rag_evaluation_combined_semantic.xlsx") as writer:
    df_cosine.to_excel(writer, sheet_name="Cosine Similarity", index=False)
    df_distilbert.to_excel(writer, sheet_name="DistilBERT Similarity", index=False)
    df_minilm.to_excel(writer, sheet_name="MiniLM Similarity", index=False)
    df_fasttext.to_excel(writer, sheet_name="FastText Similarity", index=False)

print("Evaluation completed! Results saved to rag_evaluation_combined_semantic.xlsx")

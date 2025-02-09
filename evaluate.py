import os

from ragas import SingleTurnSample
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

import pandas as pd
import asyncio
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langfuse import Langfuse

# Set Environment Variables
os.environ["LANGFUSE_SECRET_KEY"] = "secret key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "public key"

# Define RAG evaluation metrics
metrics = [faithfulness, answer_relevancy, context_utilization]

# Initialize LLM and Embeddings
llm = ChatOpenAI()
emb = OpenAIEmbeddings()


# Load dataset from CSV
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna('')  # Replace NaN with empty strings to avoid validation errors
    return df


def init_ragas_metrics(metrics, llm, embedding):
    """Initialize RAG metrics with LLM and Embeddings"""
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        metric.init(RunConfig())





async def score_row(row):
    """Scores a single row asynchronously."""
    question, contexts, answer = row["User Query"], row["Reference Text"], row["Model Response"]

    # Ensure answer is a valid string
    if not isinstance(answer, str) or answer.strip() == "":
        print(f"Skipping invalid answer for question: {question}")
        return {**row, "faithfulness": None, "answer_relevancy": None, "context_utilization": None}

    scores = {}

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[contexts] if isinstance(contexts, str) else contexts,  # Ensure it's a list
        response=answer
    )

    for metric in metrics:
        print("triggered for metric", metric, "sample", sample)
        scores[metric.name] = await metric.single_turn_ascore(sample)  # Pass Pydantic model instead of dict

    return {**row, **scores}  # Return original row with scores


async def evaluate_dataset(df):
    """Evaluates all rows concurrently and stores results."""
    tasks = [score_row(row) for _, row in df.iterrows()]  # Convert DataFrame rows to async tasks
    results = await asyncio.gather(*tasks)  # Run all tasks concurrently

    return pd.DataFrame(results)  # Convert to DataFrame


# Main Execution
if __name__ == "__main__":
    df = load_csv_data("/Users/samhithdara/PycharmProjects/pdf_reader/generated_test_data.csv")
    init_ragas_metrics(metrics, LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb))

    # Integrate Langfuse for Tracing
    langfuse = Langfuse()
    # Run evaluation asynchronously
    scores_df = asyncio.run(evaluate_dataset(df))


    # Save results to a CSV file
    output_file = "/Users/samhithdara/PycharmProjects/pdf_reader/evaluated_scor.csv"
    scores_df.to_csv(output_file, index=False)

    print(f" Evaluation complete! Scores saved to {output_file}")

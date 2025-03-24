import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import csv

# Set up Google Gemini API key
genai.configure(api_key="key")  # Replace with your actual API key

# Persistent storage location
persist_directory = "docs/chroma/"


# Function to generate embeddings using Gemini
def get_gemini_embedding(text, title="Document Chunk"):
    """Generates embeddings using Google Gemini API."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title=title
    )
    return response["embedding"]


# Wrapping Gemini embeddings in a class to work with ChromaDB
class GeminiEmbeddings:
    """Wrapper for Google Gemini embeddings to work with LangChain"""

    def embed_documents(self, texts):
        """Embeds a list of documents using Gemini API"""
        return [get_gemini_embedding(text, title=f"Chunk {i + 1}") for i, text in enumerate(texts)]

    def embed_query(self, text):
        """Embeds a single query"""
        return get_gemini_embedding(text, title="Query")


# Initialize Gemini embeddings
embedding_function = GeminiEmbeddings()


def create_vector_embed():
    """Extracts text from PDFs, splits into chunks, and stores embeddings in ChromaDB."""

    book_paths = [
        # "/Users/samhithdara/PycharmProjects/llama/docs/AlgoComp_by_Hebert S. Wilf.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/19908___Introduction to Algorithms.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/A Common-Sense Guide to Data Structures and Algorithms, Second Edition (Jay Wengrow) (Z-Library).pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/algobook-G. Pandurangan. Algorithms An Intuitive Approach.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Algorithhms 4th Edition by Robert Sedgewick, Kevin Wayne.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Algorithm Design by Jon Kleinberg, Eva Tardos.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Algorithm Design_TK.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/ALGORITHMS - ROBERT SEDGEWICK.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Algorithms and Complexity.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Algorithms by Vazirani and Dasgupta.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Algorithms for Optimization.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Anany Levitin - Introduction to The Design and Analysis of Algorithms (2nd ed).pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/Dsa.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/introduction-to-algorithms-3rd-edition-sep-2010.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/ITCS6114_2009_Introduction_to_Algorithms_Third_Ed_By_ThomasHCormen_good.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/ITCS6114_Graduate_Algorithms-JeffE-UIUC.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/SkienaTheAlgorithmDesignManual.pdf",
        "/Users/samhithdara/PycharmProjects/llama/docs/Recurrence.pdf"
    ]

    docs = []
    for path in book_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        docs.extend(text_splitter.split_documents(pages))

    # Load or create Chroma vector store
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    # Add new documents to the existing vector store
    vectordb.add_documents(docs)

    print(f"Total number of documents indexed: {vectordb._collection.count()}")


def query_vectordb(query):
    """Queries the vector database and returns the most relevant documents."""

    num_docs = 5
    total_score = 0
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    docs = vectordb.similarity_search_with_score(query, k=num_docs)

    for doc, score in docs:
        # print(f"Score: {score:.4f}")
        total_score += score  # Sum up scores

    # Compute the average score
    avg_score = total_score / num_docs if num_docs > 0 else 0
    print(docs)


    return docs,avg_score


def append_row_in_csv(csv_filename, num_docs, score1, score2, score3):
    """Reads the existing CSV, appends a new row at the end, and writes back."""

    # Read existing CSV data (if file exists)
    try:
        with open(csv_filename, mode="r", newline="") as file:
            reader = list(csv.reader(file))
    except FileNotFoundError:
        reader = []  # If file doesn't exist, start with an empty list

    # Append new row at the end
    new_row = [num_docs, score1, score2, score3]
    reader.append(new_row)  # Adds row at the last position

    # Write updated data back to the CSV file
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(reader)  # Overwrite with modified content

    print(f"Appended row to {csv_filename}")



# Uncomment to generate embeddings first
create_vector_embed()
num_docs=14
# Test query
csv_file = "retrieval_scores.csv"

_,score1 = query_vectordb("1. Solve the following recurrence relation using substitution method.")
_,score2 = query_vectordb("2. Estimate how many times faster quicksort will sort an array of one million random numbers on average than selection sort.")
_,score3 = query_vectordb("""3. Briefly write down the steps of your algorithm for below question.
1. You are given n telephone bills and m checks sent to pay the bills (n ≥ m). Assuming that telephone numbers are written on the checks, find out who failed to pay.  (For simplicity, you may also assume that only one check is written for a particular bill and that it covers the bill in full)""")
print(num_docs, score1, score2, score3)
append_row_in_csv(csv_file, num_docs, score1, score2, score3)
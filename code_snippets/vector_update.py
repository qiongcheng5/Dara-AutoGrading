# Before we are using open ai embedding model for vector embeddings that also I changed to hugging face embeddings.


from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent storage location
persist_directory = "docs/chroma/"

def create_vector_embed():
    # List of books to be embedded
    book_paths = [
        # "/Users/samhithdara/PycharmProjects/llama/docs/A Common-Sense Guide to Data Structures and Algorithms, Second Edition (Jay Wengrow) (Z-Library).pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/19908___Introduction to Algorithms.pdf",
        # "/Users/samhithdara/PycharmProjects/llama/docs/algobook-G. Pandurangan. Algorithms An Intuitive Approach.pdf",
        "/Users/samhithdara/PycharmProjects/llama/docs/AlgoComp_by_Hebert S. Wilf.pdf"
    ]

    docs = []
    for path in book_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs.extend(text_splitter.split_documents(pages))

    # Load or create Chroma vector store
    if os.path.exists(persist_directory):  # Check if the database exists
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        print("Existing database loaded.")
    else:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory
        )
        print("New database created.")

    # Add new documents to the existing vector store
    vectordb.add_documents(docs)

    print(f"Total number of documents indexed: {vectordb._collection.count()}")

def query_vectordb(query):
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    docs = vectordb.similarity_search(query, k=5)
    return docs

create_vector_embed()  # Uncomment this line to generate embeddings first
# Run the query
print(query_vectordb("1. Solve the following recurrence relation using substitution method."))

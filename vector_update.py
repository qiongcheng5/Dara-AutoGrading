import asyncio
import math
import numpy as np
import os
import kdbai_client as kdbai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_parse import LlamaParse
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.llms import LangchainLLMWrapper
import nest_asyncio
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity, BleuScore, AspectCritic  # Import SemanticSimilarity

nest_asyncio.apply()
os.environ[
    "LLAMA_CLOUD_API_KEY"] = "key"  # getpass("LlamaParse API Key: ")
os.environ[
    "OPENAI_API_KEY"] = "key"  # getpass("OpenAI API Key: ")

KDBAI_ENDPOINT = "https://cloud.kdb.ai/instance/i48orq7ae7"
KDBAI_API_KEY = "key"
create = True

session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)

# Generate test dataset
test_data_samples = []
# for query in queries:
#     response, reference = RAG(query)
#     test_data_samples.append(SingleTurnSample(user_input=query, response=response, reference=reference))


# Define the schema for the KDB.AI table
schema = [
    {"name": "document_id", "type": "str"},
    {"name": "text", "type": "str"},
    {"name": "embeddings", "type": "float32s"},
]

# Define the index parameters
indexFlat = {
    "name": "flat",
    "type": "flat",
    "column": "embeddings",
    "params": {'dims': 1536, 'metric': 'L2'},
}

# Initialize the database and table
db = session.database("default")
table_name = "LlamaParse_Table"

table = db.table(table_name)
if create:
    try:
        db.table(table_name).drop()  # Drop table if it exists
    except kdbai.KDBAIException:
        pass  # Ignore if table doesn't exist

    # Create the table with the defined schema and index
    table = db.create_table(table_name, schema, indexes=[indexFlat])
    pdf_folder = "/Users/samhithdara/PycharmProjects/llama/docs"

    # Iterate over all PDF files in the folder
    for pdf_file_name in os.listdir(pdf_folder):
        if pdf_file_name.endswith(".pdf"):
            pdf_file_path = os.path.join(pdf_folder, pdf_file_name)


            documents = SimpleDirectoryReader(pdf_file_path).load_data()
            VectorStoreIndex.from_documents(documents)

            # Initialize the embedding model (assuming you're using OpenAI embeddings)
            embedding_model = OpenAIEmbedding()


            # Generate embeddings for the documents
            for doc in documents:
                embedding = embedding_model.get_text_embedding(doc.text)
                doc.embedding = np.array(embedding, dtype='float32')  # Convert to numpy array with float32 dtype

            # Insert documents and their embeddings into KDB.AI
            vectors = []
            for doc in documents:
                embedding = np.array(doc.embedding, dtype='float32')
                vectors.append({
                    "document_id": doc.doc_id,
                    "text": doc.text,
                    "embeddings": embedding.tolist(),  # Convert numpy array to list
                })

            # Insert the data into the KDB.AI table
            try:
                batch_size = 100
                total_vectors = len(vectors)
                num_batches = math.ceil(total_vectors / batch_size)

                for i in range(num_batches):
                    batch = vectors[i * batch_size: (i + 1) * batch_size]
                    try:
                        succ = table.insert(batch)
                        print(f"Batch {i + 1}/{num_batches} inserted successfully.")
                    except Exception as e:
                        succ = "failed"
                        print(f"Error inserting batch {i + 1}/{num_batches}: {e}")
                print("Data inserted successfully", succ)
            except Exception as e:
                print(f"Error inserting data: {e}")

client = OpenAI()

def embed_query(query):
    embedding_model = OpenAIEmbedding()
    embedding = embedding_model.get_text_embedding(query)  # Ensure it's the correct method to generate embeddings
    query_embedding = np.array(embedding, dtype='float32')
    return query_embedding

def retrieve_data(query):
    query_embedding = embed_query(query)
    results = table.search(vectors={'flat': [query_embedding]})
    retrieved_data_for_RAG = []
    for index, row in results[0].iterrows():
        retrieved_data_for_RAG.append(row['text'])
    return retrieved_data_for_RAG

# global reference
def RAG(query):
    question = "You will answer this question based on the provided reference material: " + query
    messages = "Here is the provided context: " + "\n"
    results = retrieve_data(query)
    global reference
    print("retrieved context: ", results)
    if results:
        for data in results:
            messages += data + "\n"
        reference = messages
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": question},
                {"role": "context", "content": messages},
            ],
            max_tokens=300,
        )
        content = response.choices[0].message.content
    return content

query = ("Can you grade student response out of 10 - the time complexity of quick sort is O(nlog n) "
         "for the question explain time complexity of quick sort algorithm?")

# Define test data
test_data = {
    "user_input": "Explain the time complexity of QuickSort.",
    "response": RAG(query),
    "reference": reference
}

test_data_sample = SingleTurnSample(**test_data)



# Run the rest of your metrics
metric = BleuScore()
print("Bleuscore metrics:", metric.single_turn_score(test_data_sample))

# LLM metrics
correctness_critic = AspectCritic(
    name="correctness",
    llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o")),
    definition="Does the LLM-generated grade align with the reference grade and rubric?"
)

conciseness_critic = AspectCritic(
    name="conciseness",
    llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o")),
    definition="Is the feedback concise yet detailed enough to justify the grade?"
)

# Async function to run the evaluation
async def evaluate():
    correctness_score = await correctness_critic.single_turn_ascore(test_data_sample)
    conciseness_score = await conciseness_critic.single_turn_ascore(test_data_sample)
    print("Correctness Score:", correctness_score)
    print("Conciseness Score:", conciseness_score)

# Run the async function
asyncio.run(evaluate())

from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
# Initialize the semantic similarity scorer
scorer = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings()))

# Define test data
test_data_sample = SingleTurnSample(**test_data)

# Async function to run semantic similarity evaluation
async def evaluate_similarity():
    similarity_score = await scorer.single_turn_ascore(test_data_sample)
    print("Semantic Similarity Score:", similarity_score)

# Run the async function
asyncio.run(evaluate_similarity())


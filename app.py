import pandas as pd

import os
import time
import ast
import math
from typing import Any, Generator, List, Tuple, Optional
from dotenv import dotenv_values
import functools
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import gc
import json
import random


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import torch
from peft import LoraConfig, PeftModel

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.schema.retriever import BaseRetriever
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from streamlit_server_state import server_state, server_state_lock
import streamlit as st
from pprint import pprint
from google.cloud import aiplatform, bigquery, storage
from vertexai.preview.language_models import TextEmbeddingModel


CONFIG_FILE = ".env"

def upload_to_gcs(bucket_name, file_directory, file_name, folder:str=""):
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(folder+file_name)
    source_file_path = file_directory+"/"+file_name

    generation_match_precondition = 0
    uri = None
    try:
        blob.upload_from_filename(file_directory+"/"+file_name, if_generation_match=generation_match_precondition)
        uri = "gs://"+bucket_name+"/"+file_name
        print(f"File {file_name} uploaded as {uri}.")
    except:
        print(f"File {file_name} upload Fail to {bucket_name}.")
    return uri

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.llm = None

    def load_model(self, task="text-generation", temperature=0.2, repetition_penalty=1.1, return_full_text=True, max_new_tokens=1000):
        if not self.llm:
            print("Loading the model ...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            # Activate 4-bit precision base model loading
            use_4bit = True
            # Compute dtype for 4-bit base models
            bnb_4bit_compute_dtype = "float16"
            # Quantization type (fp4 or nf4)
            bnb_4bit_quant_type = "nf4"
            # Activate nested quantization for 4-bit base models (double quantization)
            use_nested_quant = False

            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

            # Check GPU compatibility with bfloat16
            if compute_dtype == torch.float16 and use_4bit:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    print("=" * 80)
                    print("GPU supports bfloat16: accelerate training with bf16=True")
                    print("=" * 80)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )
            
            model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config)
            response_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task=task,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            max_new_tokens=max_new_tokens,
            )
            self.llm = HuggingFacePipeline(pipeline=response_generation_pipeline)
            return self.llm

class Dataset:
    def __init__(self, dataset_type, db_connection:str=None, bucket_uri:str=None, create_embedding:bool=False, index_endpoint_name:str=None):
        self.type = dataset_type
        self.db_connection = db_connection
        self.bucket_uri = bucket_uri
        self.create_embedding = create_embedding
        self.chunked_documents = []
        self.index_endpoint_name=index_endpoint_name
        self.query_template = """
                SELECT distinct q.id, q.url, q.body, q.title, q.description
                FROM (SELECT * FROM `{table_name}`) AS q ORDER BY q.id
                LIMIT {limit} OFFSET {offset};
                """

    

    def _query_bq_chunks(self, max_rows: int, rows_per_chunk: int, start_chunk: int = 0) -> Generator[pd.DataFrame, Any, None]:
        client = bigquery.Client()
        table_name = self.db_connection
        for offset in range(start_chunk, max_rows, rows_per_chunk):
            query = self.query_template.format(table_name=table_name, limit=rows_per_chunk, offset=offset)
            query_job = client.query(query)
            rows = query_job.result()
            df = rows.to_dataframe()
            df["page_content"] = df.title + "\n" + df.body# + "\n" + df.description + "\n" + df.url  
            yield df

    def load_local_dataset(self, dataset_name:str="documents.csv", local_dir:str="./data") -> pd.DataFrame:
        """
        Load dataset from file_path

        Args:
            dataset_name (str, optional): Dataset name. Defaults to "documents.csv".

        Returns:
            pd.DataFrame: Dataset
        """
        print("Loading the Dataset ...")
        file_path = os.path.join(local_dir, dataset_name)
        df = pd.read_csv(file_path)
        print("Dataset Loaded in Panda DF !")
        return df
    

    
    def create_chunked_docs(self, dataset:pd.DataFrame, chunk_size:int, chunk_overlap:int) -> list:
        """
        Create chunks from the dataset

        Args:
            dataset (pd.DataFrame): Dataset
            chunk_size (int): Chunk size
            chunk_overlap (int): Chunk overlap

        Returns:
            list: List of chunks
        """
        print("Creating the Chunks ...")
        text_chunks = DataFrameLoader(
            dataset, page_content_column="body"
        ).load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
        )

        loader = DataFrameLoader(dataset, page_content_column="body")
        documents_loaded = loader.load()
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
        docs_splitted = text_splitter.split_documents(documents_loaded) # Return the list of documents chunked by chunk size

        for doc in docs_splitted:
            title = doc.metadata["title"]
            task_type = "RETRIEVAL_DOCUMENT",
            description = doc.metadata["description"]
            content = doc.page_content
            url = doc.metadata["url"]
            final_content = f"TITLE: {title}\DESCRIPTION: {description}\BODY: {content}\nURL: {url}"
            doc.page_content = final_content
        print("Chunks created !")
        return text_chunks   

class Documents:

    class Embedding:
        def __init__(self, embedding_model="textembedding-gecko"):
            self.model = self._get_embedding_model(embedding_model)
            self.type = None
            self.model_generation = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")


        def _get_embedding_model(self, embedding_model="textembedding-gecko"):
            if embedding_model == "textembedding-gecko":
                self.type = "Google"
                return VertexAIEmbeddings(model_name='textembedding-gecko@latest', task_type="retrieval_document", project=server_state.config["PROJECT_ID"], location=server_state.config["REGION"])
            self.type = "Hugging_Face"
            return HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Define an embedding method that uses the model
        def encode_texts_to_embeddings(self, sentences: List[str]) -> List[Optional[List[float]]]:
            try:
                embeddings = self.model_generation.get_embeddings(sentences)
                return [embedding.values for embedding in embeddings]
            except Exception:
                return [None for _ in range(len(sentences))]

        # Generator function to yield batches of sentences
        def generate_batches(self, sentences: List[str], batch_size: int) -> Generator[List[str], None, None]:
            for i in range(0, len(sentences), batch_size):
                yield sentences[i : i + batch_size]


        def encode_text_to_embedding_batched(self, sentences: List[str], api_calls_per_second: int = 10, batch_size: int = 5) -> Tuple[List[bool], np.ndarray]:

            embeddings_list: List[List[float]] = []

            # Prepare the batches using a generator
            batches = self.generate_batches(sentences, batch_size)
            seconds_per_job = 1 / api_calls_per_second

            with ThreadPoolExecutor() as executor:
                futures = []
                for batch in tqdm(
                    batches, total=math.ceil(len(sentences) / batch_size), position=0
                ):
                    futures.append(
                        executor.submit(functools.partial(self.encode_texts_to_embeddings), batch)
                    )
                    time.sleep(seconds_per_job)

                for future in futures:
                    embeddings_list.extend(future.result())
            is_successful = [
                embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
            ]
            embeddings_list_successful = np.squeeze(
                np.stack([embedding for embedding in embeddings_list if embedding is not None])
            )
            return is_successful, embeddings_list_successful

        def generate_embeddings(self, dataset:Dataset) -> str:
            print("Let's Generate Embeddings !")

            # Create temporary file to write embeddings to
            embeddings_file_path = Path(tempfile.mkdtemp())
            chunk_path = embeddings_file_path.joinpath(
               f"{embeddings_file_path.stem}.json"
            )

            print(f"Embeddings directory: {embeddings_file_path}")
            df = next(dataset._query_bq_chunks(max_rows=1000, rows_per_chunk=1000))
            is_successful, question_chunk_embeddings = self.encode_text_to_embedding_batched(
                sentences=df.page_content.tolist()
            )
            print("Done with Embedding Generation !")
            with open(chunk_path, "a") as f:
                id_chunk = df.id

                embeddings_formatted = [
                    json.dumps(
                        {
                            "id": str(id),
                            "embedding": [str(value) for value in embedding],
                        }
                    )
                    + "\n"
                    for id, embedding in zip(id_chunk[is_successful], question_chunk_embeddings)
                ]
                f.writelines(embeddings_formatted)
            embeddings_dimension = len(question_chunk_embeddings[0])

            questions = df.page_content.tolist()
            questions = np.array(questions)[is_successful]
            question_index = random.randint(0, 99)
            
            question = df.loc[df['id'] == question_index]
            print(f"Query question = {question}")

            print("Embedding Page Content:")
            print(question_chunk_embeddings[question_index])

            print("Transpose Matrix:")
            print(question_chunk_embeddings.T)

            scores = np.dot(question_chunk_embeddings[question_index], question_chunk_embeddings.T)

            print("Scores")
            print(scores)


            for index, (question, score) in enumerate(
                sorted(zip(questions, scores), key=lambda x: x[1], reverse=True)[:20]
            ):
                print(f"\tBest Dot Product:{index}: {question}: {score}")

            del df
            gc.collect()
            return embeddings_file_path, embeddings_dimension

    
    class VectorStore:
        
        def __init__(self, embedding, dataset):
            self.vector_type = dataset.type
            self.vector_db_path = dataset.db_connection
            self.store = self._create_vector_store(embedding, dataset) 

        def _load_local_vector_db(self, embedding_model, dataset) -> FAISS:

            """
            Create or get vector store

            Args:

                chunks (list): List of chunks

            Returns:
                FAISS: Vector store
            """
            if not os.path.exists(self.vector_db_path):
                if dataset is None:
                    dataset = Dataset("local")
                df = dataset.load_local_dataset(
                    server_state.config["LOCAL_DATASET_FILENAME"], 
                    server_state.config["LOCAL_DATASET_DIRECTORY"])
                chunked_docs = dataset.create_chunked_docs(df, 1000, 0)
                vectorstore = FAISS.from_documents(
                    chunked_docs, embedding_model
                )
                vectorstore.save_local(self.vector_db_path)
            else:
                vectorstore = FAISS.load_local(self.vector_db_path, embedding_model)
            return vectorstore 
        
        def _create_online_vector_index(self, embeddding_file_path:str, embedding_dimensions:int, bucket_uri:str, folder:str):
            remote_folder = "gs://"+bucket_uri+"/"+folder

            display_name = "gcp_documentation"
            description = "Page content from GCP documentation"

            # Create Vector Search Index
            print("Creating the Vector Search index...")
            tree_ah_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=display_name,
                contents_delta_uri=remote_folder,
                dimensions=embedding_dimensions,
                approximate_neighbors_count=150,
                distance_measure_type="DOT_PRODUCT_DISTANCE",
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=80,
                description=description,
            )
            index_resource_name = tree_ah_index.resource_name
            tree_ah_index = aiplatform.MatchingEngineIndex(index_name=index_resource_name)

            # Create Index Endpoint
            print("Creating the Index endpoint...")
            my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=display_name,
                description=description,
                public_endpoint_enabled=True
            )

            # Deploy vector search index on endpoint
            print("Deploying the Vector Search index on the endpoint...")
            deployed_index_id = self.vector_db_path
            my_index_endpoint = my_index_endpoint.deploy_index(
                index=tree_ah_index, deployed_index_id=deployed_index_id
            )
            my_index_endpoint.deployed_indexes
            return my_index_endpoint


        def _load_online_vector_db(self, embedding, dataset):
            print("Online DB: For Vector Search")

            if dataset.create_embedding is True:
                # Create Embeddings
                embeddings_files_path, embedding_dimensions = embedding.generate_embeddings(dataset)

                # Create Vector Search Index
                self.vector_db_path = dataset.db_connection+"_vector"
                for embeddings_file_name in os.listdir(embeddings_files_path):
                    file_uploaded_uri = upload_to_gcs(dataset.bucket_uri, str(embeddings_files_path), embeddings_file_name, "embeddings/")
                
                index_endpoint = self._create_online_vector_index(embeddings_files_path, embedding_dimensions, dataset.bucket_uri, "embeddings/")
                print(embeddings_files_path)
                print(index_endpoint)
            else:
                index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=dataset.index_endpoint_name)
            print("We have a Vector DB online !")
            retriever_r = CustomRetriever(vector_search_endpoint=index_endpoint, embedding=embedding, dataset=dataset)
            return retriever_r

        def _create_vector_store(self, embedding, dataset):
            if dataset.type == "local":
                self.vector_type = "FAISS"
                return self._load_local_vector_db(embedding.model, dataset)
            self.vector_type = "VECTOR_SEARCH"
            return self._load_online_vector_db(embedding, dataset)
    
    def __init__(self, dataset):
        self.vector_type = dataset.type
        self.embedding = self.Embedding("textembedding-gecko")
        self.vector = self.VectorStore(self.embedding, dataset)
    
class CustomRetriever(BaseRetriever):
    vector_search_endpoint:aiplatform.MatchingEngineIndexEndpoint
    embedding:Documents.Embedding
    dataset:Dataset

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        result_docs = list()
        k_documents = 1
        idx_neighbors = []
        client = bigquery.Client()
        query_template = """
                SELECT distinct q.url, q.body
                FROM (SELECT distinct id, url, body FROM `{table_name}` WHERE id IN UNNEST ({docs_neighbors})) AS q
                LIMIT {limit};
                """

        query_embeddings = self.embedding.encode_texts_to_embeddings(sentences=[query])
        response = self.vector_search_endpoint.find_neighbors(
            deployed_index_id=self.Dataset.vector.vector_db_path, #"thisisatestthomaslemoulleclala"
            queries=query_embeddings,
            num_neighbors=k_documents,
        )
        for match_index, neighbor in enumerate(response[0]):
            idx_neighbors.append(int(neighbor.id))

        query = query_template.format(table_name=self.dataset.db_connection, limit=k_documents, docs_neighbors=idx_neighbors)
        query_job = client.query(query)
        rows = query_job.result()
        for row in rows:
            result_docs.append(Document(page_content=row[1], metadata={"url": row[0]}))
        return result_docs

    def as_retriever(self) -> BaseRetriever:
        return self

def get_conversation_chain(llm:HuggingFacePipeline, vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Get the chatbot conversation chain

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def run_app(system_message_prompt, human_message_prompt):
    print("Run App")
    response = None

    with st.spinner("Processing..."):
        user_question = st.chat_input("Ask your question")
        if user_question:
            response = st.session_state.conversation({"question": user_question})
        st.session_state.conversation = get_conversation_chain(
        server_state.model.llm, server_state.documents.vector.store, system_message_prompt, human_message_prompt
    )

    if response is not None:
        st.session_state.chat_history += response["chat_history"]
        length_chat = 0
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0: # User
                st.chat_message("user").markdown(message.content)
            else:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    if (i == len(st.session_state.chat_history)-1): #Last message: let's pretend the AI is typing
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in message.content.split('\n\n'):
                            full_response += chunk + " "
                            time.sleep(0.2)
                            # Add a blinking cursor to simulate typing
                            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    message_placeholder.markdown(message.content)

def initialize_server():

    ### Loading Environnemnt variables to build the Classes
    server_state.config = None
    server_state.config = dotenv_values(CONFIG_FILE)
    server_state.config["CREATE_EMBEDDINGS"] = True if server_state.config["CREATE_EMBEDDINGS"] == "True" else False
    if server_state.config is None:
        print("Please Load a server configuration file...")
        return -1

    ### Initialise GCP Client APIs
    aiplatform.init(project=server_state.config["PROJECT_ID"], location=server_state.config["REGION"])

    ### Setup the Model
    if "model" not in server_state:
        print("No model loaded in the server")
        server_state.model = Model(server_state.config["MODEL_ANSWER"])
        server_state.model.llm = server_state.model.load_model()
    
    ### Setup the Documents Store
    if "documents" not in server_state:
        print("No vector store loaded in the server")
        server_state.documents = Documents(Dataset(server_state.config["DB_TYPE"], server_state.config["DB_CONNECTION_STRING"], 
            server_state.config["ONLINE_DATASET_BUCKET"], server_state.config["CREATE_EMBEDDINGS"], 
            server_state.config["VECTOR_SEARCH_ENDPOINT_NAME"]))
    ### Check Status before Loading the app
    if "model" in server_state and "documents" in server_state:
        server_state.init_server = True
        print("Server Initialisation is done")

def initialize_session():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    initialize_session()
    if "init_server" not in server_state:
        server_state.init_server = None
        initialize_server()
    
    if server_state.init_server is not None:
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with responding to questions about the documentation of Google Cloud.

            You should never answer a question with a question, and you should always respond with the most relevant documentation page.

            Do not answer questions that are not about the Google Cloud platform.
            
            Include the relevant documentation URL or code snippets and examples.

            Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
            {context}
            """
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

        st.set_page_config(
            page_title="GCP Assistant",
            page_icon=":rocket:"
        )

        st.title("GCP Assistant")
        st.subheader("Ask me your GCP questions !")
        st.markdown(
            f"""
            This chatbot powered by {server_state.config["MODEL_ANSWER"]} was created to answer questions about the GCP documentation.
            Ask a question and the chatbot will respond with the most relevant page of the documentation.
            """
        )
        st.image(server_state.config["BANNER_IMG"])

        run_app(system_message_prompt, human_message_prompt)
    else:
        st.title("GCP Assistant")
        st.subheader("We are loading the model "+server_state.config["MODEL_ANSWER"]+" in GPU memory")
        st.markdown(
            """
            Please wait...
            """
        )

if __name__ == "__main__":
    main()
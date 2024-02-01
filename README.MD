# GCP Assistant with RAG and Open Source

Through this project you will be able to implement and deploy an end to end RAG system. 
After deploying the server, you will have the possibility to chat with an assistant that will build answers based on a specific context (E.g: Documentation pages).

Launching such prototype helps to quickly get your hands on the real challenges of a RAG System. The current stage of this project is a good baseline to **start optimizing** the RAG answers **accuracy**, the **performance** of the application, the **reliability** of the overall pipeline. 

```
![GCP Assistant Screenshot](https://github.com/thomaslemoullec/rag-gcp-mistral/blob/main/assets/gcp_assistant.png)
```

By default, this project is showcasing an assistant that provides answers based on GCP documentation. Here is the ***logical flow*** behind the project:

```
![RAG System Logical Flow](https://github.com/thomaslemoullec/rag-gcp-mistral/blob/main/assets/logical_flow.png)
```

This project is modular as its main purpose is **prototyping** and **experimentation**: 

 - You have the flexibility to chose a **Local mode** (CSV + FAISS Vector DB) or an **Online Mode** (Big Query, Cloud Storage, Vector Search DB). 
 - If you want to change Agent documentation scope, you can just change the root sitemap.xml and the System prompt. More about this configuration setup later...


# Files
For easier prototyping, the application is distributed in 2 simple files that can be deployed in the same server : 

 1. ***build_dataset.py*** : covers the Data ingestion and Data processing for Web pages 
 2. **app.py** : Cover the web server, dataset and db interactions, and langchain logic implementation

Other files:

 - ***./streamlit/config.toml*** : Describe the streamlit server configuration
 - ***./data/*** : Folder that contains two example files: One for the URL extraction and the other one for the web pages extraction
 - ***requirements.txt*** : Define the packages to install

# Getting Started

### Setup the environment
Pull the public git repository  

> **git clone https://github.com/thomaslemoullec/rag-gcp-mistral.git**

Make sure your Dev environment is well authenticated ([See ADC Documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc))

Install the python packages

> **pip install -r requirements.txt**

If `which streamlit` is not returning the path, check your .bashrc/.zshrc files and update the PATH with the location of pip installation. 

### Build the Dataset

Define some environment variables in a `.env` in the root folder:

    PROJECT_ID="accelerator-ai"
    REGION="europe-west4"
    SITEMAP_URL="https://cloud.google.com/tpu/sitemap.xml"
    
    #Local DB Setup
    LOCAL_DATASET_DIRECTORY="./data/"
    LOCAL_DATASET_FILENAME="documents.csv"
    LOCAL_SITEMAP_FILENAME="sitemap_index.csv"
    
    #Remote DB Setup
    ### ONLINE DATASET SETUP ###
    ONLINE_DATASET_BUCKET="rag-dataset-artefacts"
    DATABASE="webpages_documentation"
    DATABASE_REGION="EU"
    TABLE_NAME_SITEMAP="sitemap"
    TABLE_NAME_DOCUMENTS="documents"

 - **Sitemap_URL**: Here you define the sitemap URL for the website that will be indexed. For testing purposes I am only using a subset of GCP documentation with the TPU documentation. 
 - **Local DB Setup**: Here you define how the data will be stored locally
 - **Online DB Setup**: Here you define the Big Query setup for storing the Website content and index 
In the code you can define with a boolean if you want CSV to be uploaded to Cloud Storage on top of the Big Query insertion


Run the script:
> **python build_dataset.py**

### Run the Server
[**Optional**] - Install CUDA Drivers, check [Hardware and Driver requirements here](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)

Define more environment variables in the `.env` file, you will add this variables: 

    DB_TYPE="online"
    DB_CONNECTION_STRING="${PROJECT_ID}.${DATABASE}.${TABLE_NAME_DOCUMENTS}"
    CREATE_EMBEDDINGS="True"
    VECTOR_DB_ENDPOINT="projects/PROJECT_ID/locations/europe-west4/indexEndpoints/RANDOM_ID"

 - **DB_Type** online will be using Big Query as database
 - **DB_CONNECTION_STRING** is the Table containing the Web pages content
 - **CREATE_EMBEDDINGS** set to true is quite an heavy process as it will download all the webpages content and generate embeddings for them, then it will load this embeddings in a vector search DB which can take some time if it needs to be created
 - If you define **CREATE_EMBEDDINGS** to "**False**", you can also define the **VECTOR_DB_ENDPOINT** with the endpoint that you previously created (check in the console)

When using `**local**` instead of `**online**` the software will be using FAISS vector DB locally and the CSV files previously stored on the disk in `LOCAL_DATASET_DIRECTORY`
If you want to use an offline / local setup for the database and the Vector store, you can replace with:

    DB_TYPE="local"
    DB_CONNECTION_STRING="./db"

**Finally** : Run the Web server with the environnement variables

> **streamlit run app.py**

## Cloud Architecture

Here is a more advanced and scalable architecture deployment for this project:

```
![GCP Cloud Architecture](https://github.com/thomaslemoullec/rag-gcp-mistral/blob/main/assets/cloud_architecture_rag.png)
```


## Roadmap

 1. Build a Microservice approach with Container -> 3 docker files (Data pipeline, Web Server, Model Endpoint)
 2. Optimize Latency (Context, Prompt, Retrieval method and DB, Model Loading parameters, Kernel optimization, Hardware setup, parallelism)
 3. Optimize Rag system with Advanced technic
 4. Optimize Data Source, Data Enrichment, Meta Data, Filters

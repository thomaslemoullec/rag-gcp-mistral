# Need to build :
- Check if we need to combine chunked documents for better retrieval
- Create modular VectorDB Class (Local and Vector Search): if vector search need to create Embedding Class without Langchain
- Create embeddings into Json : https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/matching_engine/sdk_matching_engine_create_stack_overflow_embeddings_vertex.ipynb
- Store Embeddings in Cloud Storage
- Vector search as custom retriever (https://python.langchain.com/docs/modules/data_connection/retrievers/) + https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore. + https://stackoverflow.com/questions/77156069/using-langchain-with-custom-retrievers / If Vector Search does not work, implement with pg_vector in AlloyDB
- More complex Chain with Double LLMs : https://medium.com/@jerome.o.diaz/langchain-conversational-retrieval-chain-how-does-it-work-bb2d71cbb665 and https://docs.google.com/presentation/d/16V-BooaHoiFTl5R7mr3qg4dC0g_WJCHMtO_kX489qNw/edit#slide=id.g29af1e91813_0_564
- Env Variables: Define local variables and the server run configuration
- Explore filtering and other features from Vector Search : https://python.langchain.com/docs/modules/data_connection/vectorstores/ 
- Update Template and store in Class Template
- Different Classes in different file
- Deploy the app on GCE
- Deploy the app on GKE
- Build the cloud platform: Workflows: Build dataset and insert into Big Query, Firestore


## Optimization: 
* Reliability: 
- Store buffer memory in redis : https://github.com/langchain-ai/langchain/pull/2122
- Multiple instances
* Performance: 
- Loading the model
- Multiple GPUs
- Query and Chains
- Larger Index
- Split the model and use TGI / vLLM
* Cost: 




##### .ENV Example

PROJECT_ID="PROJECT_ID"
REGION="europe-west4"
MODEL_ANSWER="HuggingFaceH4/zephyr-7b-beta"
MODEL_ICON=""
BANNER_IMG="https://www.cxtoday.com/wp-content/uploads/2020/04/Google-Cloud-Rapid-Response-Virtual-Agent-UC-Today-Google-Contact-Center-AI-.png"
MODEL_QUERY="HuggingFaceH4/zephyr-7b-beta"

## SITEMAP to Fetch
SITEMAP_URL="https://cloud.google.com/tpu/sitemap.xml" # Use this Sitemap to only index TPU documentation, for faster Data generation. Or 
#SITEMAP_URL="https://cloud.google.com/sitemap.xml" # Complete GCP Sitemap
#SITEMAP_URL="https://cloud.google.com/docs/sitemap.xml" # Complete GCP Docs Sitemap

### LOCAL DATASET SETUP ###
LOCAL_DATASET_DIRECTORY="./data/"
LOCAL_DATASET_FILENAME="documents.csv"
LOCAL_SITEMAP_FILENAME="sitemap_index.csv"

### ONLINE DATASET SETUP ###
ONLINE_DATASET_BUCKET="BUCKET_NAME"
DATABASE="DATASET_BQ"
DATABASE_REGION="EU"
TABLE_NAME_SITEMAP="TABLE_SITEMAP_BQ"
TABLE_NAME_DOCUMENTS="TABLE_DOC_BQ"

### DEFINE DB SETUP
#DB_TYPE="local"
#DB_CONNECTION_STRING="./db"
# IF ONLINE:
DB_TYPE="online"
DB_CONNECTION_STRING="${PROJECT_ID}.${DATABASE}.${TABLE_NAME_DOCUMENTS}"
CREATE_EMBEDDINGS="True"
VECTOR_SEARCH_ENDPOINT_NAME="VECTOR_SEARCH_ENDPOINT"
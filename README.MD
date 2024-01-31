

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

### 
Run on a server : sudo apt update & sudo apt install python3-pip https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/index.html and https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
#### Commands: 
sudo systemctl stop google-cloud-ops-agent
sudo apt update & sudo apt install python3-pip 
sudo apt-get install linux-headers-$(uname -r)
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

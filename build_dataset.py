import time
import pandas as pd
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata
import csv
import sys
import threading
from google.cloud import bigquery, storage

lock = threading.Lock()

from dotenv import dotenv_values

CONFIG_FILE = ".env"
config = dotenv_values(CONFIG_FILE)

import re

def extract_en_urls(text) -> list:
    return ([word for word in text.split() if word.startswith(("https://", "http://")) and ("?hl=" not in word or word.endswith("?hl=en"))])

def extract_all_urls(text) -> list:
    url_regex = r'\b(?:https?://|www\.)\S+(?<!\?hl=|\?hl=en)\b'
    return re.findall(url_regex, text)

def insert_urls_into_list(text, en=True) -> list:
    if en is not True:
        return extract_all_urls(text)
    return extract_en_urls(text)

def get_urls_from_sitemap(resource_url: str, k=None) -> list:
    """
    Recovers the sitemap through Trafilatura
    """
    urls = []
    site_map_html = fetch_url(resource_url)
    extract_html = extract(site_map_html)
    if extract_html is not None:
        urls = insert_urls_into_list(extract_html)
    if k is not None:
        return urls[:k]
    return urls

def process_url(url, data, idx):
    html = fetch_url(url)
    if html:
        body = extract(html)
        metadata = extract_metadata(html)
        title = metadata.title if metadata else ""
        description = metadata.description if metadata else ""
        d = {
            'id': idx,
            'url': url,
            "body": body,
            "title": title,
            "description": description
        }
        with lock:
            data.append(d)

def fetch_data_urls(sitemap_list:str) -> list:
    data = []
    threads = []
    for idx, url in enumerate(sitemap_list):
        thread = threading.Thread(target=process_url, args=(url, data, idx))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return data

def extract_urls_not_xml(sitemap_list:list) -> list:
    result = []
    for source_url, listed_urls in sitemap_list.items():
        for url in listed_urls:
            if not url.endswith(".xml") and "docs" in url:
                result.append(url)
    return result

def extract_urls_from_csv(data_path:str="./data/", file_name:str="sitemap"):
    docs_urls = []
    
    with open(data_path+file_name, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            url_string = row["Listed_URLS"]
            urls = [url.strip() for url in url_string.split(',')]
            for url in urls:
                if "docs" in url:
                    docs_urls.append(url)
    return docs_urls


def create_csv_index_sitemap(sitemap: list, data_path:str="./data/", file_name:str="sitemap"):
    with open(data_path+file_name, 'w', newline='') as csvfile:
        # Create CSV writer object
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Source_URL', 'Listed_URLS'])
        # Iterate through nested dictionary
        for source_url, listed_urls in sitemap.items():
            # Write each row
            csv_writer.writerow([source_url, ', '.join(listed_urls)])


def create_sitemap_dict(url, sitemap_dict=None):
    if sitemap_dict is None:
        sitemap_dict = {}
    sitemap_dict[url] = get_urls_from_sitemap(url)
    for link in sitemap_dict[url]:
        if link.endswith(".xml"):
            create_sitemap_dict(link, sitemap_dict)
    return sitemap_dict

def fetch_sitemap(url, sitemap_dict=None) -> list:
    sitemap_dict = create_sitemap_dict(url)
    return sitemap_dict

def create_csv_documents(data:list, data_path="./data", dataset_name="documents"):
    print("Loading Documents Data in CSV")
    print(data)
    csv_file_path = data_path+"/"+dataset_name
    fieldnames = ['id', 'url', 'body', 'title', 'description']

    # Write data to CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        for data_doc in data:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data_doc)

def extract_urls(sitemap_list=None, data_path:str="./data/", file_name:str="sitemap", loading_csv=False):
    if loading_csv is not True:
        return extract_urls_not_xml(sitemap_list)
    
    return extract_urls_from_csv(data_path, file_name)

def insert_into_bq(data:list, table_id:str):
    print("Inserting Data into:")
    qc = bigquery.Client(project=config["PROJECT_ID"])
    dataset = qc.dataset(config["DATABASE"])
    table = dataset.table(config["TABLE_NAME_DOCUMENTS"])
    table_nm = qc.get_table(table)
    errors = qc.insert_rows_json(
    table_nm, data)
    if errors:
        print("Encountered errors while inserting rows: {}".format(errors))

def load_csv_into_bq(table_id:str, schema, dataset_uri:str):
    print("Loading Dataset to Big Query")

    # Construct a BigQuery client object.
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(schema=schema,
        skip_leading_rows=1,
        # The source format defaults to CSV, so the line below is optional.
        source_format=bigquery.SourceFormat.CSV,
    )

    load_job = client.load_table_from_uri(
        dataset_uri, table_id, job_config=job_config
    )  # Make an API request.

    load_job.result()  # Waits for the job to complete.

def upload_dataset_bucket(bucket_name, dataset_dir, dataset_file_name, folder:str=""):
    print("Uploading Dataset to Cloud Storage")
    timestamp = time.time()
    timestamp = str(timestamp)

    storage_client = storage.Client()
    bucket_file_name = timestamp+"-"+dataset_file_name
    source_file_path = dataset_dir+dataset_file_name

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(folder+bucket_file_name)

    generation_match_precondition = 0
    uri = None
    try:
        blob.upload_from_filename(source_file_path, if_generation_match=generation_match_precondition)
        uri = "gs://"+bucket_name+"/"+folder+bucket_file_name
        print(f"File {source_file_path} uploaded as {uri}.")
    except:
        print(f"File {source_file_path} upload Fail to {bucket_file_name}.")
    return uri

def create_table(table_id:str, schema) -> str:
    client = bigquery.Client()
    table = bigquery.Table(table_id, schema=schema)
    try:
        table = client.create_table(table)  # Make an API request.
        print("Creating Table...")
        print(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )
    except:
        print("Information : Table probably exist")
    table_id = table.project+"."+table.dataset_id+"."+table.table_id
    print("Table ID:")
    print(table_id)
    print("Table object:")
    print(table)
    return table

def create_dataset(dataset_name:str, dataset_region:str="EU"):

    # Construct a BigQuery client object.
    client = bigquery.Client()

    # Construct a full Dataset object to send to the API.
    dataset = bigquery.Dataset(dataset_name)

    # TODO(developer): Specify the geographic location where the dataset should reside.
    dataset.location = dataset_region

    # Send the dataset to the API for creation, with an explicit timeout.
    # Raises google.api_core.exceptions.Conflict if the Dataset already
    # exists within the project.
    try:
        dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
        print("Created dataset {}.{}".format(client.project, dataset.dataset_id))
    except:
        print("Dataset already exist")
        dataset = client.get_dataset(dataset_name)
    return dataset

def process_sitemaps(table_id_sitemap:str, sitemap_url:str, local_dir:str, local_sitemap_file:str, online_bucket:str, insert_bq:bool=True) -> list:
    # Step 0 : Create Sitemap index
    sitemap_list = fetch_sitemap(sitemap_url)
    create_csv_index_sitemap(sitemap_list, local_dir, local_sitemap_file) # create index as csv file


    # Upload sitemap CSV to cloud storage
    dataset_sitemap_uri = upload_dataset_bucket(online_bucket, local_dir, local_sitemap_file, "sitemaps/")

    if insert_bq == True:
        schema_sitemap=[
                bigquery.SchemaField("Source_URL", "STRING", "NULLABLE"),
                bigquery.SchemaField("Listed_URLS", "STRING", "NULLABLE"),
            ]
        load_csv_into_bq(table_id_sitemap, schema_sitemap, dataset_sitemap_uri)
    return sitemap_list

def process_documents(sitemap_list:list, table_id_documents:str, local_dir, local_sitemap_file:str, local_dataset_file:str, online_bucket:str, insert_bq:bool=True, upload_csv:bool=False):
    # Step 1 : Get URLs from Sitemap
    urls_to_fetch = extract_urls(sitemap_list, local_dir, local_sitemap_file, False)

    # Step 2 : Fetch URLs and Create Documents CSV
    data = fetch_data_urls(urls_to_fetch)

    if insert_bq == True:
        print("Lets insert documents in BQ")
        insert_into_bq(data, table_id_documents)
    if upload_csv == True:
        create_csv_documents(data, local_dir, local_dataset_file)
        dataset_sitemap_uri = upload_dataset_bucket(online_bucket, local_dir, local_dataset_file, "documents/")

def initialize_database(dataset_id:str, region:str, table_id_documents:str, table_id_sitemap:str):
    print("Initializing Databases...")
    create_dataset(dataset_id, config["DATABASE_REGION"])
    schema_documents=[
                bigquery.SchemaField("id", "INT64", "NULLABLE"),
                bigquery.SchemaField("url", "STRING", "NULLABLE"),
                bigquery.SchemaField("body", "STRING", "NULLABLE"),
                bigquery.SchemaField("title", "STRING", "NULLABLE"),
                bigquery.SchemaField("description", "STRING", "NULLABLE")
            ]
    table_id_documents = create_table(table_id_documents, schema_documents)
    schema_sitemap=[
                bigquery.SchemaField("Source_URL", "STRING", "NULLABLE"),
                bigquery.SchemaField("Listed_URLS", "STRING", "NULLABLE"),
            ]
    table_id_sitemap = create_table(table_id_sitemap, schema_sitemap)
    time.sleep(5)
    return table_id_documents, table_id_sitemap



def main():
    csv.field_size_limit(sys.maxsize)
    insert_bq = True
    dataset_id = config["PROJECT_ID"]+"."+config["DATABASE"]
    table_id_sitemap = dataset_id+"."+config["TABLE_NAME_SITEMAP"]
    table_id_documents = dataset_id+"."+config["TABLE_NAME_DOCUMENTS"]
    if insert_bq is True:
        table_id_documents, table_id_sitemap = initialize_database(dataset_id, config["DATABASE_REGION"], table_id_documents, table_id_sitemap)
    sitemaps_list = process_sitemaps(table_id_sitemap, config["SITEMAP_URL"], config["LOCAL_DATASET_DIRECTORY"], config["LOCAL_SITEMAP_FILENAME"], config["ONLINE_DATASET_BUCKET"], insert_bq)
    process_documents(sitemaps_list, table_id_documents, config["LOCAL_DATASET_DIRECTORY"], config["LOCAL_SITEMAP_FILENAME"], config["LOCAL_DATASET_FILENAME"], config["ONLINE_DATASET_BUCKET"], insert_bq, True)
    return 0

if __name__ == "__main__":
    main()
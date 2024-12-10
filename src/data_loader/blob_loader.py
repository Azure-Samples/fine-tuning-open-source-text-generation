from azure.storage.blob import BlobServiceClient
import os
from io import StringIO
import pandas as pd
import boto3


class BlobStorageHandler:
    def __init__(self, connection_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )

    def upload_data(self, file_path, blob_name):
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data)
            print(f"File {file_path} uploaded to blob {blob_name}.")
        except Exception as e:
            print(f"Error uploading file: {e}")

    def download_data(self, blob_name, download_file_path):
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Blob {blob_name} downloaded to {download_file_path}.")
        except Exception as e:
            print(f"Error reading blob: {e}")

    def read_blob(self, blob_name):
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()
            content = blob_data.decode("utf-8")

            if blob_name.endswith(".csv"):
                # Handle CSV file
                data = pd.read_csv(StringIO(content))
            elif blob_name.endswith(".txt"):
                # Handle TXT file
                data = pd.read_csv(StringIO(content), delimiter="\t")
            else:
                print(f"Unsupported file type for blob: {blob_name}")
                return None

            return data
        except Exception as e:
            print(f"Error reading blob: {e}")
            return None


if __name__ == "__main__":
    # Example usage:
    # Load the workspace from the saved config file
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())

    # READING FROM AZURE BLOB STORAGE

    # Retrieve the connection string for the Azure Blob Storage
    connection_string = os.environ.get("CONNECTION_STR")
    container_name = "mlsandbox-container"
    handler = BlobStorageHandler(connection_string, container_name)
    # handler.upload_data("path/to/local/file.txt", "blob_name.txt")
    # handler.download_data("test.txt", "path/to/local/file.txt")
    content = handler.read_blob("test.csv")

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os

from azure.storage.blob import BlobServiceClient


def download_az_file_blobs(blob_names_dict, az_container_name="myconedesx7"):
    conn_str = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT')};"
        f"AccountKey={os.getenv('AZURE_STORAGE_KEY')};"
        f"EndpointSuffix={os.getenv('ENDPOINT_SUFFIX')}"
    )
    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=conn_str
    )
    for az_blob_name, local_file_path in blob_names_dict.items():
        blob_client = blob_service_client.get_blob_client(
            container=az_container_name, blob=az_blob_name
        )
        with open(local_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

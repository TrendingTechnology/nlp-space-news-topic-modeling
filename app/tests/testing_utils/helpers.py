#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from io import StringIO

import pandas as pd
from azure.storage.blob import BlobServiceClient


def load_az_blob_data(
    az_storage_container_name="myconedesx7",
    az_blob_name="blobedesz39",
    n_rows=None,
    index_col=None,
):
    conn_str = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT')};"
        f"AccountKey={os.getenv('AZURE_STORAGE_KEY')};"
        f"EndpointSuffix={os.getenv('ENDPOINT_SUFFIX')}"
    )
    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=conn_str
    )
    blob_client = blob_service_client.get_blob_client(
        container=az_storage_container_name, blob=az_blob_name
    )
    download_stream = blob_client.download_blob()
    blobstring = download_stream.content_as_text()
    df_new = pd.read_csv(
        StringIO(blobstring), nrows=n_rows, index_col=index_col
    )
    return df_new

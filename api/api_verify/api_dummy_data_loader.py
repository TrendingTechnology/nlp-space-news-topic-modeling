#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import random
from io import StringIO

import pandas as pd
from azure.storage.blob import BlobServiceClient


# load_az_blob_data() taken from api/tests/testing_utils/data_loader.py
def load_az_blob_data(
    az_storage_container_name="myconedesx7",
    az_blob_name="blobedesz42",
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


def save_file_get_endpoint_dict(df, dummy_data_filename, data_dir):
    dummy_data_filepath = os.path.join(data_dir, dummy_data_filename)
    if not os.path.exists(dummy_data_filepath):
        df.to_csv(dummy_data_filepath, index=False)
    file_dict = {
        "data_filepath": (
            f"{dummy_data_filepath}",
            open(f"{dummy_data_filepath}", "rb"),
            "text/csv",
        ),
    }
    return file_dict


def get_dummy_invalid_url_data(dummy_data_filepath):
    with open(dummy_data_filepath) as json_file:
        d_observations_urls_raw = json.load(json_file)
    d_observations_urls = []
    d_observations_urls_all = []
    for d_observations_url_raw in d_observations_urls_raw:
        d_observation_url_formatted = {
            "data": [
                {
                    k: v
                    for k, v in d_observations_url_raw.items()
                    if k not in ["status_code", "msg"]
                }
            ],
            "status_code": d_observations_url_raw["status_code"],
            "msg": d_observations_url_raw["msg"],
        }
        d_observations_urls.append(d_observation_url_formatted)
        d_observation_all_urls_formatted = {
            k: v
            for k, v in d_observations_url_raw.items()
            if k not in ["status_code", "msg"]
        }
        d_observations_urls_all.append(d_observation_all_urls_formatted)
    return [d_observations_urls, d_observations_urls_all]


def make_random_sentence():
    nouns = ["puppy", "car", "rabbit", "girl", "monkey"]
    verbs = ["runs", "hits", "jumps", "drives", "barfs"]
    adv = ["crazily", "dutifully", "foolishly", "merrily", "occasionally"]
    adj = ["adorable.", "clueless.", "dirty.", "odd.", "stupid."]

    random_entry = lambda x: x[random.randrange(len(x))]
    return " ".join(
        [
            random_entry(nouns),
            random_entry(verbs),
            random_entry(adv),
            random_entry(adj),
        ]
    )


def generate_dummy_invalid_csv_data():
    return {
        # Length of text is below acceptable number of characters
        "dummy_uploadcsv1.csv": {
            "data": pd.DataFrame.from_records(
                [
                    {
                        "url": f"dummy_url_{str(k+1)}",
                        "text": make_random_sentence(),
                    }
                    for k in range(10)
                ]
            ),
            "status_code": 200,
            "error_msg": (
                "All News article texts are not long enough. "
                "Need more than 108 characters."
            ),
        },
        # Missing required text field
        "dummy_uploadcsv2.csv": {
            "data": pd.DataFrame.from_records(
                [{"url": f"dummy_url_{str(k+1)}"} for k in range(10)]
            ),
            "status_code": 200,
            "error_msg": (
                "File dummy_uploadcsv2.csv missing required column 'text'"
            ),
        },
        # Text field is present but contains empty string (not long enough)
        "dummy_uploadcsv3.csv": {
            "data": pd.DataFrame.from_records(
                [
                    {"url": f"dummy_url_{str(k+1)}", "text": ""}
                    for k in range(10)
                ]
            ),
            "status_code": 200,
            "error_msg": (
                "All News article texts are not long enough. "
                "Need more than 108 characters."
            ),
        },
    }

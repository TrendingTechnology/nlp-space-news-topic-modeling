#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import pytest
from api_helpers.api_scraping_helpers import download_az_file_blobs
from tests.testing_utils import data_loader as dl

dict_blob_to_local = {
    "training_residuals_file_path": {
        "blob_name": "blobedesz39",
        "local": "training_residuals_for_api_webpage.csv",
        "nrows": None,
    },
    "local_file_path": {
        "blob_name": "blobedesz42",
        "local": "guardian_3.csv",
        "nrows": 5,
    },
    "topic_names_filepath": {
        "blob_name": "blobedesz40",
        "local": "nlp_topic_names.csv",
        "nrows": None,
    },
    "topic_residuals_filepath": {
        "blob_name": "blobedesz43",
        "local": "nlp_topic_residuals.csv",
        "nrows": None,
    },
}


@pytest.fixture
def az_blob_data_tmp_file_path(tmp_path):
    """Use pandas to read data from blob storage and export to local file."""
    d = tmp_path / "data"
    d.mkdir()
    d_file_paths = {}
    for local_file_path, v in dict_blob_to_local.items():
        file_path = d / v["local"]
        if not os.path.exists(file_path):
            df = dl.load_az_blob_data(
                az_blob_name=v["blob_name"], n_rows=v["nrows"]
            )
            df.to_csv(file_path, index=False)
        d_file_paths[local_file_path] = file_path
    return d_file_paths


@pytest.fixture
def get_data_from_blob_storage():
    """Get data from Azure blob storage."""
    pipe_filepath = "data/nlp_pipe.joblib"
    azure_blob_file_dict = {"blobedesz41": pipe_filepath}
    if not os.path.exists(pipe_filepath):
        download_az_file_blobs(azure_blob_file_dict)
    return {"pipe": pipe_filepath}

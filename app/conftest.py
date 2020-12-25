#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import pytest
from tests.testing_utils import helpers as dh

from scraping_helpers import download_az_file_blobs


@pytest.fixture
def unseen_data_tmp_file_path(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    local_file_path = d / "guardian_3.csv"
    df_unseen_data = dh.load_az_blob_data(n_rows=5)
    df_unseen_data.to_csv(local_file_path, index=False)
    return local_file_path


@pytest.fixture
def get_data_from_blob_storage():
    pipe_filepath = "data/nlp_pipe.joblib"
    topic_names_filepath = "data/nlp_topic_names.csv"
    azure_blob_file_dict = {
        "blobedesz40": topic_names_filepath,
        "blobedesz41": pipe_filepath,
    }
    if not os.path.exists(pipe_filepath):
        download_az_file_blobs(azure_blob_file_dict)
    return {"pipe": topic_names_filepath, "topic_names": topic_names_filepath}

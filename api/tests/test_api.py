#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import pytest
from starlette.testclient import TestClient

from api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert type(response.history) == list
    assert len(response.history) == 0
    assert response.status_code == 200
    assert response.url == "http://testserver/"
    assert response.json()["message"] == "Hello World"


def test_docs_redirect():
    response = client.get("/docs")
    assert type(response.history) == list
    assert len(response.history) == 0
    assert response.status_code == 200
    assert response.url == "http://testserver/docs"


@pytest.mark.scrapingtest
def test_predictor_multi_input_predict_from_url():
    test_url = (
        "https://www.theguardian.com/science/2019/dec/09/european-space-"
        "agency-to-launch-clearspace-1-space-debris-collector-in-2025"
    )
    json_input = [{"url": test_url}]
    response = client.post(
        "/predict",
        json=json_input,
    )
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 1

    assert records[0]["topic"] == "Scientific Research about Dark Matter"


def test_predictor_multi_input_predict_from_file(
    az_blob_data_tmp_file_path, get_data_from_blob_storage
):
    for local_file in [
        "local_file_path",
        "topic_names_filepath",
        "topic_residuals_filepath",
    ]:
        assert os.path.exists(az_blob_data_tmp_file_path[local_file])
    assert ["pipe"] == list(get_data_from_blob_storage.keys())
    with az_blob_data_tmp_file_path["local_file_path"].open("rb") as f:
        filebody = f.read()
    local_file_path = str(az_blob_data_tmp_file_path["local_file_path"])
    response = client.post(
        "/uploadcsv",
        files={"data_filepath": (local_file_path, filebody)},
    )
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 5

    assert records[0]["topic"] == "Scientific Research about Dark Matter"
    second_unseen_topic = "Sun's influence on life across the Solar System"
    assert records[1]["topic"] == second_unseen_topic
    assert records[2]["topic"] == "Search for E.T. life"
    assert records[3]["topic"] == "Space Debris from Satellites"
    assert records[4]["topic"] == "Black Holes"

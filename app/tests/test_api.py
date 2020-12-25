#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from api import app
from starlette.testclient import TestClient

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert type(response.history) == list
    assert len(response.history) == 0
    assert response.status_code == 200
    assert response.url == "http://testserver/"


def test_docs_redirect():
    response = client.get("/docs")
    assert type(response.history) == list
    assert response.status_code == 404
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

    assert records[0]["best"] == "Space Debris from Satellites"


def test_predictor_multi_input_predict_from_file(
    unseen_data_tmp_file_path, get_data_from_blob_storage
):
    assert ["pipe", "topic_names"] == list(get_data_from_blob_storage.keys())
    with unseen_data_tmp_file_path.open("rb") as f:
        filebody = f.read()
    response = client.post(
        "/uploadcsv",
        files={"data_filepath": (str(unseen_data_tmp_file_path), filebody)},
    )
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 5

    for record in records[:3]:
        assert record["best"] == "Space Debris from Satellites"
    assert records[3]["best"] == (
        "Sun's influence on life across the Solar System"
    )
    assert records[4]["best"] == "ISS updates"

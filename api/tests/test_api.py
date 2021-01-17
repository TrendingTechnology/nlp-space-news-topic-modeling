#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from datetime import date

import pytest
from apit import app
from httpx import AsyncClient
from starlette.testclient import TestClient
from tests.testing_utils import unit_testing_funcs as utf

client = TestClient(app)

# dates of unseen news articles that will be used to determine returned keys
beginning_date = date(2019, 11, 2)  # date of first unseen news article
ending_date = date(2020, 2, 27)  # date of last unseen news article
n_days = (ending_date - beginning_date).days


def test_root():
    response = client.get("/")
    assert type(response.history) == list
    assert len(response.history) == 0
    assert response.status_code == 200
    assert response.url == "http://testserver/"
    assert response.json()["message"] == "Hello World!"


@pytest.mark.asyncio
@pytest.mark.scrapingtest
async def test_async_predictor_multi_input_predict_from_url():
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        test_urls = [
            (
                "https://www.theguardian.com/science/2019/dec/09/european-"
                "space-agency-to-launch-clearspace-1-space-debris-collector-"
                "in-2025"
            ),
            (
                "https://www.theguardian.com/science/2019/dec/19/true-"
                "meanings-of-words-of-emotion-get-lost-in-translation-"
                "study-finds"
            ),
        ]
        json_input = [{"url": test_url} for test_url in test_urls]
        response = await ac.post(
            "/api/v1/topics/predict",
            json=json_input,
        )
    assert response.status_code == 200
    assert response.url == "http://testserver/api/v1/topics/predict"
    records = response.json()
    assert len(records) == 2

    assert records[0]["topic"] == "Space Debris from Satellites"
    assert records[1]["topic"] == "Topic 0"

    utf.check_max_date_range(records, n_days)


def test_docs_redirect():
    response = client.get("/docs")
    assert type(response.history) == list
    assert len(response.history) == 0
    assert response.status_code == 200
    assert response.url == "http://testserver/docs"


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
        "/api/v1/topics/uploadcsv",
        files={"data_filepath": (local_file_path, filebody)},
    )
    assert response.status_code == 200
    assert response.url == "http://testserver/api/v1/topics/uploadcsv"
    records = response.json()
    assert len(records) == 5

    assert records[0]["topic"] == "Scientific Research about Dark Matter"
    second_unseen_topic = "Sun's influence on life across the Solar System"
    assert records[1]["topic"] == second_unseen_topic
    assert records[2]["topic"] == "Search for E.T. life"
    assert records[3]["topic"] == "Space Debris from Satellites"
    assert records[4]["topic"] == "Black Holes"

    utf.check_max_date_range(records, n_days)

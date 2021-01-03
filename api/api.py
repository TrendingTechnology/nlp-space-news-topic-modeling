#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""Space News Article Topic Predictor API."""


import os
from typing import Any, List, Mapping

import api_helpers.api_scraping_helpers as apih
import api_helpers.api_topic_predictor as tp
import api_helpers.api_utility_helpers as lh
import pandas as pd
from api_helpers.api_loading_helpers import handle_upload_file
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

PROJ_ROOT_DIR = os.getcwd()
jinja2_templates_dir = os.path.join(PROJ_ROOT_DIR, "templates")

pipe_filepath = "data/nlp_pipe.joblib"
topic_names_filepath = "data/nlp_topic_names.csv"
topic_residuals_filepath = "data/nlp_topic_residuals.csv"
urls_to_read = 1
n_topics_wanted = 35
n_top_words = 10
test_url = (
    "https://www.theguardian.com/science/2019/dec/09/european-space-"
    "agency-to-launch-clearspace-1-space-debris-collector-in-2025"
)
azure_blob_file_dict = {
    "blobedesz40": topic_names_filepath,
    "blobedesz41": pipe_filepath,
    "blobedesz43": topic_residuals_filepath,
}
unseen_urls_filepath = os.path.join(
    PROJ_ROOT_DIR, "api_helpers", "api_unseen_article_urls.yml"
)
api_image_specs_filepath = os.path.join(
    PROJ_ROOT_DIR, "api_helpers", "api_image_specs.yml"
)

# Data download from blob storage, if not found locally
if not os.path.exists(pipe_filepath):
    apih.download_az_file_blobs(azure_blob_file_dict)

pipe = load(pipe_filepath)
df_named_topics = pd.read_csv(topic_names_filepath, index_col="topic_num")
df_residuals_new = pd.read_csv(
    topic_residuals_filepath, parse_dates=["start_date", "end_date"]
)
description = lh.get_description_html(
    jinja2_templates_dir,
    "linked_image_grid.j2",
    unseen_urls_filepath,
    api_image_specs_filepath,
    6,
)
# print(description)


class Url(BaseModel):
    """Parse and validate news article url"""

    url: str

    class Config:
        schema_extra = {"example": {"url": test_url}}


Urls = List[Url]

app = FastAPI(
    title="Space News Article Topic Predictor API",
    description=description,
    docs_url="/",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


@app.post("/predict")
async def predict_from_url(
    urls: Urls,
) -> Mapping[str, Any]:
    """Predict topic from url(s).

    Parameters
    ----------
    urls : Dict[str: str]
    - news article url to be retrieved

    Example Inputs
    --------------
    ```
    [
        {"url": "https://www.google.com"},
        {"url": "https://www.yahoo.com"}
    ]
    ```

    ```
    [
        {"url": "https://www.google.com"}
    ]
    ```

    Returns
    -------
    news article topic, text and other predicted/metadata : Dict[str: str]
    - predicted topic and article text, eg.
      ```
      {
          "url": "https://www.google.com",
          "text": "abcd"
          "topic_num": "32",
          "topic": 32,
          "best": "Topic Name Here"
      }
      ```
    """
    guardian_urls = [dict(url)["url"] for url in urls]
    df_new = apih.scrape_new_articles(guardian_urls)
    response_dict = tp.generate_response(
        df_new,
        pipe,
        n_top_words,
        n_topics_wanted,
        df_named_topics,
        df_residuals_new,
    )
    return response_dict


@app.post("/uploadcsv")
async def predict_from_file(
    data_filepath: UploadFile = File(...),
) -> Mapping[str, Any]:
    """Predict topic from a CSV file of news article url & text.

    Parameters
    ----------
    file : Dict[str: float]
    - csv file with following for each news article
      - url
      - full text

    Example Inputs
    --------------
    ```
    url,text
    https://www.google.com,abcd
    https://www.yahoo.com,efgh
    ```

    ```
    url,text
    https://www.google.com,abcd
    ```

    Returns
    -------
    news article topic, text and other predicted/metadata : Dict[str: str]
    - predicted topic and article text, eg.
      ```
      {
          "text": "abcd"
          "topic_num": "32",
          "topic": 32,
          "best": "Topic Name Here"
      }
      ```
    """
    # print(data_filepath.file)
    # df_new = pd.read_csv(Path("data") / Path(data_filepath.filename))
    df_new = handle_upload_file(data_filepath, pd.read_csv)
    response_dict = tp.generate_response(
        df_new,
        pipe,
        n_top_words,
        n_topics_wanted,
        df_named_topics,
        df_residuals_new,
    )
    return response_dict

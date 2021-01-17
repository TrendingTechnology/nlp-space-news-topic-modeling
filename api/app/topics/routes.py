#!/usr/bin/python3
# -*- coding: utf-8 -*-


import asyncio
import os
from datetime import date
from typing import Any, List, Mapping

import aiohttp
import api_helpers.api_scraping_helpers as apih
import api_helpers.api_topic_predictor as tp
import numpy as np
import pandas as pd
from api_helpers.api_loading_helpers import handle_upload_file
from bs4 import BeautifulSoup
from fastapi import APIRouter, File, UploadFile
from joblib import load
from pydantic import BaseModel

router = APIRouter()

# dates of unseen news articles that will be used to determine returned keys
beginning_date = date(2019, 11, 2)  # date of first unseen news article
ending_date = date(2020, 2, 27)  # date of last unseen news article
pipe_filepath = "data/nlp_pipe.joblib"
topic_names_filepath = "data/nlp_topic_names.csv"
topic_residuals_filepath = "data/nlp_topic_residuals.csv"
topic_training_residuals_statistics_filepath = (
    "data/training_residuals_for_api_webpage.csv"
)
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
    "blobedesz39": topic_training_residuals_statistics_filepath,
}

# Data download from blob storage, if not found locally
if not os.path.exists(pipe_filepath):
    apih.download_az_file_blobs(azure_blob_file_dict)

pipe = load(pipe_filepath)
df_named_topics = pd.read_csv(topic_names_filepath, index_col="topic_num")
df_residuals_new = pd.read_csv(
    topic_residuals_filepath, parse_dates=["start_date", "end_date"]
)
n_days = (ending_date - beginning_date).days
# print(n_days)


class Url(BaseModel):
    """Parse and validate news article url"""

    url: str

    class Config:
        schema_extra = {"example": {"url": test_url}}


Urls = List[Url]
zip_texts = dict()


class RetrieveTextForUrl(BaseModel):

    url = str

    @classmethod
    async def retrieve_text(cls, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.read()
                # print(content)
                try:
                    soup = BeautifulSoup(content.decode("utf-8"), "lxml")
                    # print(soup.prettify())
                except Exception as e:
                    print(f"Experienced error {str(e)} when scraping {url}")
                    text = np.nan
                else:
                    text = apih.get_guardian_text_from_soup(soup)
                # print(text)
        zip_texts.update({url: text})


@router.post("/predict")
async def predict_from_url(
    urls: Urls,
) -> Mapping[str, Any]:
    """Predict topic from url(s).

    Parameters
    ----------
    urls : Dict[str: str]
      - news article url(s) to be retrieved

    Example Inputs
    --------------
    - multiple news article urls
        ```
        [
            {"url": "https://www.google.com"},
            {"url": "https://www.yahoo.com"}
        ]
        ```

    - single news article url
        ```
        [
            {"url": "https://www.google.com"}
        ]
        ```

    Returns
    -------
    news article topic, text and other predicted/metadata : Dict[str: str]
    - predicted topic, metadata and related quantities
      - for single news article url
        - `url`
          - news article url
        - `date`
          - news article publication date
        - `year`
          - year of publication
        - `week_of_month`
          - week of month in which article was published
        - `weekday`
          - day of week on which article was published
        - `month`
          - month of year in which article was published
        - `text`
          - retrieved text of news article
        - `topic_num`
          - arbitrarily assigned topic internal number
        - `topic`
          - predicted name of topic
        - `term`
          - top 10 NLP tokens for topic, found during training
        - `term_weight`
          - weighted TFIDF term weights, found during training
        ```
        {
            "url": "https://www.google.com",
            "date": "1900-01-01T00:00:00",
            "year": "1900",
            "week_of_month": 1,
            "weekday": "Monday",
            "month": "Jan",
            "text": "<full article text>",
            "topic_num": <internally assigned topic number from 1 to 35>,
            "topic": "<topic name>",
            "term": [
                "abc",
                "defg",
                .
                .
                .
            ],
            "term_weight": [
                0.100,
                0.200,
                .
                .
                .
            ]
        }
        ```
      - additional keys, if all acceptable news article urls
        - `entity`
          - up to 10 most frequently occurring organizations in article
        - `entity_count`
          - respective number of occurrences of organizations in article
        - `resid_min`
          - min. residual (Frobenius norm) between NMF approx. and true data
        - `resid_max`
          - max. residual between NMF approximation and true data
        - `resid_perc25`
          - 25th percentile of residual between NMF approx. and true data
        - `resid_perc75`
          - 75th percentile of residual between NMF approx. and true data
        ```
        {
            "entity": [
                "abc",
                "defg",
                .
                .
                .
            ],
            "entity_count": [
                2,
                2,
                2,
                .
                .
                .
            ],
            "resid_min": 0.1,
            "resid_max": 0.2,
            "resid_perc25": 0.12
            "resid_perc75": 0.18
        }
        ```
    """
    guardian_urls = [dict(url)["url"] for url in urls]
    # print(guardian_urls)
    tasks = [
        asyncio.create_task(RetrieveTextForUrl.retrieve_text(guardian_url))
        for guardian_url in guardian_urls
    ]
    _ = await asyncio.gather(*tasks)
    zip_texts_cleaned = {k: v for k, v in zip_texts.items() if type(v) is str}
    # print(zip_texts_cleaned)
    df_new = (
        pd.DataFrame.from_dict(zip_texts_cleaned, orient="index")
        .reset_index()
        .rename(columns={"index": "url", 0: "text"})
    )
    # print(df_new)
    response_dict = tp.generate_response(
        df_new,
        pipe,
        n_top_words,
        n_topics_wanted,
        df_named_topics,
        df_residuals_new,
        n_days,
    )
    return response_dict


@router.post("/uploadcsv")
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
     - predicted topic, metadata and related quantities
       - for single news article text
         - `url`
           - news article url
         - `date`
           - news article publication date
         - `year`
           - year of publication
         - `week_of_month`
           - week of month in which article was published
         - `weekday`
           - day of week on which article was published
         - `month`
           - month of year in which article was published
         - `text`
           - retrieved text of news article
         - `topic_num`
           - arbitrarily assigned topic internal number
         - `topic`
           - predicted name of topic
         - `term`
           - top 10 NLP tokens for topic, found during training
         - `term_weight`
           - weighted TFIDF term weights, found during training
         ```
         {
             "url": "https://www.google.com",
             "date": "1900-01-01T00:00:00",
             "year": "1900",
             "week_of_month": 1,
             "weekday": "Monday",
             "month": "Jan",
             "text": "<full article text>",
             "topic_num": <internally assigned topic number from 1 to 35>,
             "topic": "<topic name>",
             "term": [
                 "abc",
                 "defg",
                 .
                 .
                 .
             ],
             "term_weight": [
                 0.100,
                 0.200,
                 .
                 .
                 .
             ]
         }
         ```
       - additional keys, if for all acceptable news article texts
         - `entity`
           - up to 10 most frequently occurring organizations in article
         - `entity_count`
           - respective number of occurrences of organizations in article
         - `resid_min`
           - min. residual (Frobenius norm) between NMF approx. and true data
         - `resid_max`
           - max. residual between NMF approximation and true data
         - `resid_perc25`
           - 25th percentile of residual between NMF approx. and true data
         - `resid_perc75`
           - 75th percentile of residual between NMF approx. and true data
         ```
         {
             "entity": [
                 "abc",
                 "defg",
                 .
                 .
                 .
             ],
             "entity_count": [
                 2,
                 2,
                 2,
                 .
                 .
                 .
             ],
             "resid_min": 0.1,
             "resid_max": 0.2,
             "resid_perc25": 0.12
             "resid_perc75": 0.18
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
        n_days,
    )
    return response_dict

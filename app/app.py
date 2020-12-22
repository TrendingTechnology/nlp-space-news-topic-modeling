#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""Space News Article Topic Predictor API."""


from typing import List, Mapping, Union

import pandas as pd
import topic_predictor as tp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

description = f"""
{tp.create_description_header_html()}

{tp.create_header_image_html()[0]}

{tp.create_header_image_html()[1]}

{tp.create_acceptable_urls_markdown()}
"""
pipe_filepath = "data/nlp_pipe.joblib"
topic_names_filepath = "data/nlp_topic_names.csv"
urls_to_read = 1
n_topics_wanted = 35
n_top_words = 10

test_url = (
    "https://www.theguardian.com/science/2019/dec/09/european-space-"
    "agency-to-launch-clearspace-1-space-debris-collector-in-2025"
)

pipe = load(pipe_filepath)
df_named_topics = pd.read_csv(topic_names_filepath, index_col="topic_num")


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
) -> Mapping[str, Union[str, int]]:
    """Predict topic from url(s).

    Parameters
    ----------
    urls : Dict[str: str]
    - news article url to be retrieved

    Example Inputs
    --------------
    - [
          {"url": "https://www.google.com"},
          {"url": "https://www.yahoo.com"}
      ]
    - [
          {"url": "https://www.google.com"}
      ]

    Returns
    -------
    news article topic and text : Dict[str: str]
    - predicted topic and article text, eg. {
          "url": "https://www.google.com",
          "text": "abcd"
          "topic_num": "32",
          "topic": 32,
          "best": "Topic Name Here"
      }
    """
    guardian_urls = [dict(url)["url"] for url in urls]
    df_new = tp.scrape_new_articles(guardian_urls)
    # print(df_new)
    df_predicted_new_topics = tp.make_predictions(
        df_new, pipe, n_top_words, n_topics_wanted, df_named_topics
    )
    # for k, row in df_predicted_new_topics.iterrows():
    #     print(f"Row={k}, URL={row['url']}, Topic={row['best']}\n")
    return df_predicted_new_topics.to_dict("records")


@app.post("/uploadcsv")
async def predict_from_file(
    data_filepath: UploadFile = File(...),
) -> Mapping[str, Mapping[str, float]]:
    """Predict topic from a CSV file of news article url & text.

    Parameters
    ----------
    file : Dict[str: float]
    - csv file with following for each news article
      - url
      - full text

    Example Inputs
    --------------
    url,text
    https://www.google.com,abcd
    https://www.yahoo.com,efgh

    url,text
    https://www.google.com,abcd

    Returns
    -------
    news article topic and text : Dict[str: str]
    - predicted topic and article text, eg. {
          "text": "abcd"
          "topic_num": "32",
          "topic": 32,
          "best": "Topic Name Here"
      }
    """
    df_new = pd.read_csv(data_filepath.file)
    # print(df_new)
    df_predicted_new_topics = tp.make_predictions(
        df_new, pipe, n_top_words, n_topics_wanted, df_named_topics
    )
    # for k, row in df_predicted_new_topics.iterrows():
    #     print(f"Row={k}, URL={row['url']}, Topic={row['best']}\n")
    return df_predicted_new_topics.to_dict("records")

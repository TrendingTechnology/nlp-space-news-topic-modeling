#!/usr/bin/python3
# -*- coding: utf-8 -*-


import asyncio
import os
import re
from datetime import date
from typing import Any, List, Mapping

import aiohttp
import api_helpers.api_scraping_helpers as apih
import api_helpers.api_topic_predictor as tp
import api_helpers.api_utility_helpers as apiuh
import numpy as np
import pandas as pd
import peewee as pw
from api_helpers.api_loading_helpers import handle_upload_file
from bs4 import BeautifulSoup
from fastapi import APIRouter, File, HTTPException, UploadFile
from joblib import load
from pydantic import BaseModel, HttpUrl, validator

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
min_training_chars = 135
news_article_length_allowance_factor = 0.80
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

PROJ_ROOT_DIR = os.getcwd()
data_dir = os.path.join(PROJ_ROOT_DIR, "data")
db_name = os.path.join(data_dir, "predictions.db")

# DB = pw.SqliteDatabase(db_name)

# class Prediction(pw.Model):
#     url = pw.TextField()
#     date = pw.DateTimeField()
#     year = pw.TextField()
#     week_of_month = pw.IntegerField()
#     weekday = pw.TextField()
#     month = pw.TextField()
#     topic_num = pw.IntegerField()
#     topic = pw.TextField()
#     resid = pw.FloatField()
#     valid_resid_distribution = pw.BooleanField()
#     term = pw.TextField()
#     term_weight = pw.TextField()
#     entity = pw.TextField()
#     entity_count = pw.TextField()
#     valid_residual = pw.BooleanField()
#     valid_term_weight = pw.BooleanField()
#     valid_entity_count = pw.BooleanField()
#     resid_min = pw.FloatField()
#     resid_max = pw.FloatField()
#     resid_perc25 = pw.FloatField()
#     resid_perc75 = pw.FloatField()

#     class Meta:
#         database = DB


# DB.create_tables([Prediction], safe=True)


class Url(BaseModel):
    """Parse and validate news article url"""

    url: HttpUrl

    @validator("url")
    def validate_url(cls, v):
        errors = []
        if "theguardian.com" not in v:
            errors.append("URL not from theguardian.com")

        year = ""
        month = ""
        day = 0
        month_regex_pattern = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
        mregex = r"/(\d{4})/(" + "|".join(month_regex_pattern) + r")/(\d{2})/"
        try:
            year, month, day = list(
                re.findall(
                    mregex,
                    v,
                )[0]
            )
        except IndexError as e:
            print(
                (
                    f"Error validating Pydantic HttpUrl - msg={str(e)} - {v} "
                    f"missing date info. Setting placeholder value(s)."
                )
            )
            if not year:
                year = "None"
            if not month:
                month = "None"
            if not day:
                day = -999
        # print(year, month, day, type(year), type(month), type(day))

        if int(day) not in list(range(1, 31 + 1)):
            errors.append(
                (
                    f"Invalid value provided for day: {day}. Allowed values "
                    "are: 1-31"
                )
            )
        acceptable_years = ["2019", "2020"]
        if year not in acceptable_years:
            errors.append(
                (
                    f"Invalid value provided for year: {year}. "
                    f"Allowed values are: ({', '.join(acceptable_years)})"
                )
            )
        acceptable_months = ["nov", "dec", "jan", "feb"]
        if month not in acceptable_months:
            errors.append(
                (
                    f"Invalid value provided for month: {month}. "
                    f"Allowed values are: ({', '.join(acceptable_months)})"
                )
            )
        assert not errors, f"{','.join(errors)}"
        return v

    class Config:
        schema_extra = {"example": {"url": test_url}}


Urls = List[Url]
zip_texts = dict()


class RetrieveTextForUrl(BaseModel):

    url = HttpUrl

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
        # Prediction,
        # DB,
        n_days,
    )
    return response_dict


@router.post("/uploadcsv")
async def predict_from_file(
    data_filepath: UploadFile = File(...),
) -> Mapping[str, Any]:
    # print(data_filepath.file)
    # df_new = pd.read_csv(Path("data") / Path(data_filepath.filename))
    df_new = handle_upload_file(data_filepath, pd.read_csv)

    # Sanity checks
    error = ""
    if "text" not in list(df_new):
        fname = os.path.basename(data_filepath.filename)
        error = f"File {fname} missing required column 'text'"
        if error:
            error_msg = {"detail": [{"loc": ["body"], "msg": error}]}
            return error_msg
    df_new["text"] = df_new["text"].fillna("")

    # Find texts that are not long enough - placeholder content will be added
    min_reqd_length = int(
        min_training_chars * news_article_length_allowance_factor
    )
    too_short_mask = df_new["text"].str.len() <= min_reqd_length
    mask = df_new["text"].str.len() > min_reqd_length
    df_new_too_short = df_new.loc[too_short_mask].reset_index(drop=True)
    df_new = df_new.loc[mask].reset_index(drop=True)
    if df_new.empty:
        error = (
            "All News article texts are not long enough. "
            f"Need more than {min_reqd_length} characters."
        )
        if error:
            error_msg = {"detail": [{"loc": ["body"], "msg": error}]}
            return error_msg

    response_dict = tp.generate_response(
        df_new,
        pipe,
        n_top_words,
        n_topics_wanted,
        df_named_topics,
        df_residuals_new,
        # Prediction,
        # DB,
        n_days,
        df_new_too_short,
    )
    return response_dict


predict_from_url.__doc__ = apiuh.get_url_doc()
predict_from_file.__doc__ = apiuh.get_csv_doc()

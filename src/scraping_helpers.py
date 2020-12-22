#!/usr/bin/python3
# -*- coding: utf-8 -*-


import re

import numpy as np
import pandas as pd
from newspaper import fulltext


def get_guardian_text_from_soup(soup):
    """Get Guardian text from soup object"""
    mydiv = soup.find("div", {"class": "article-body-commercial-selector"})
    # print(mydiv)
    if not mydiv:
        mydiv = soup.find("div", {"class": "content__article-body"})
    unwanted_tweets = mydiv.findAll(
        "figure", {"class": "element element-tweet"}
    )
    for unwanted in unwanted_tweets:
        unwanted.extract()
    unwanted_images = mydiv.findAll(
        "figure", {"class": "element element-embed"}
    )
    for unwanted in unwanted_images:
        unwanted.extract()
    unwanted_images2 = mydiv.findAll(
        "figure",
        {
            "class": (
                "element element-image "
                "img--landscape fig--narrow-caption fig--has-shares"
            )
        },
    )
    for unwanted in unwanted_images2:
        unwanted.extract()
    all_text = str(mydiv.text).replace("\n", "")
    art_text = all_text.split("Topics")[0]
    # print(art_text)
    return art_text


def get_hubble_text_from_soup(soup):
    """Get Hubble text from soup object"""
    mydiv = soup.find(
        "div", {"class": "square-img hide-subhead news-listing__details"}
    )
    all_text = (
        mydiv.text.strip()
        .strip()
        .replace("\n", "")
        .split("Research Box Title")[-1]
    )
    art_text = re.split("Credits|Keywords", all_text)[0]
    if not art_text:
        soup_div_text = soup.find(
            "div", {"class": "page-intro__main col-sm-12"}
        ).text
        art_text = "Abstract: " + str(soup_div_text).strip()
    # print(art_text)
    return art_text


def get_space_text_from_soup(soup, page_response):
    """Get Space.com text from soup object"""
    auth_cont = soup.find("span", {"class": "no-wrap by-author"})
    if auth_cont:
        author = str(auth_cont.text)
    else:
        author = "Person"
    if "Tariq Malik" not in author:
        art_text = (
            fulltext(page_response.text)
            .split("Video - ")[0]
            .split("Follow ")[0]
            .replace("\n", " ")
        )
        date = soup.find("meta", {"name": "pub_date"}).get("content")
    else:
        art_text = np.nan
        date = np.nan
    # print(f"Author: {author}, URL: {link}\n{art_text}\n\n")
    return art_text, date


def get_nytimes_text_from_soup(soup):
    """Get NY Times text from soup object"""
    art_text_paragraph = []
    art_body = soup.find("section", {"name": "articleBody"})
    for p in art_body.find_all("p"):
        art_text_paragraph.append(str(p.text).strip().replace("\n", ""))
        # print(p.text)
    art_text = " ".join(art_text_paragraph)
    date = soup.find("time").get("datetime")
    return art_text, date


def save_dflist_hdfs(dflist, file_name_ptrn="d:/temp/data_{:02}.h5", **kwarg):
    for i, df in enumerate(dflist):
        df.to_hdf(
            file_name_ptrn.format(i + 1), "df{:02d}".format(i + 1), **kwarg
        )
    return len(dflist)


def append_datetime_attrs(df, date_col, publication):
    """Append datetime columns to DataFrame"""
    utc_setting = True if publication == "nytimes" else None
    df[date_col] = pd.to_datetime(df[date_col], utc=utc_setting)
    L = [
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "quarter",
    ]
    df = df.join(
        pd.concat((getattr(df[date_col].dt, i).rename(i) for i in L), axis=1)
    )
    return df

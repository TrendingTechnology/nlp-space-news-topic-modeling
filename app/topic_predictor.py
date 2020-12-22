#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from time import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from processing_helpers import process_text
from scraping_helpers import get_guardian_text_from_soup


def scrape_new_articles(urls):
    l_texts = {}
    for k, link in enumerate(urls):
        print(f"Scraping article number {k+1}, Link: {link}", end="... ")
        # print(site, link)
        start_time = time()
        r_session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        r_session.mount("http://", HTTPAdapter(max_retries=retries))
        try:
            page_response = r_session.get(link, timeout=5)
        except Exception as ex:
            print(f"{ex} Error connecting to {link}")
        else:
            try:
                soup = BeautifulSoup(page_response.content, "lxml")
                # print(soup.prettify())
            except Exception as e:
                print(f"Experienced error {str(e)} when scraping {link}")
                text = np.nan
            else:
                text = get_guardian_text_from_soup(soup)
        scrape_minutes, scrape_seconds = divmod(time() - start_time, 60)
        print(
            f"Scraping time: {int(scrape_minutes):d} minutes, "
            f"{scrape_seconds:.2f} seconds"
        )
        l_texts[link] = [text]
    df = pd.DataFrame.from_dict(l_texts, orient="index").reset_index()
    df.rename(columns={"index": "url", 0: "text"}, inplace=True)
    return df


def get_top_words_per_topic(row, n_top_words=5):
    return row.nlargest(n_top_words).index.tolist()


def make_predictions(df, pipe, n_top_words, n_topics_wanted, df_named_topics):
    df["processed_text"] = df["text"].apply(process_text)

    # Transform unseen texts with the trained ML pipeline
    doc_topic = pipe.transform(df["processed_text"])

    topic_words = pd.DataFrame(
        pipe.named_steps["nmf"].components_,
        index=[str(k) for k in range(n_topics_wanted)],
        columns=pipe.named_steps["vectorizer"].get_feature_names(),
    )
    topic_df = (
        pd.DataFrame(
            topic_words.apply(
                lambda x: get_top_words_per_topic(x, n_top_words), axis=1
            ).tolist(),
            index=topic_words.index,
        )
        .reset_index()
        .rename(columns={"index": "topic"})
        .assign(topic_num=range(n_topics_wanted))
    )
    # for k, v in topic_df.iterrows():
    #     print(k, ",".join(v[1:-1]))

    df_temp = (
        pd.DataFrame(doc_topic).idxmax(axis=1).rename("topic_num").to_frame()
    )

    merged_topic = df_temp.merge(topic_df, on="topic_num", how="left").assign(
        url=df["url"].tolist()
    )
    df_topics = df.merge(merged_topic, on="url", how="left").astype(
        {"topic_num": int}
    )
    df_topics_new = df_topics[
        ["url", "text"] + ["topic_num", "topic"] + list(range(n_top_words))
    ].merge(
        df_named_topics.reset_index()[["topic_num", "best"]], on="topic_num"
    )[
        ["url", "text", "topic_num", "topic", "best"]
    ]
    return df_topics_new


def create_acceptable_urls_markdown():
    base_url = "https://www.theguardian.com/science"
    guardian_urls = [
        (
            "2019/dec/09/european-space-agency-to-launch-clearspace-1-space-"
            "debris-collector-in-2025"
        ),
        (
            "2019/nov/04/nasa-voyager-2-sends-back-first-signal-from-"
            "interstellar-space"
        ),
        (
            "2019/dec/12/spacewatch-esa-awards-first-junk-clean-up-contract-"
            "clearspace"
        ),
        "2019/nov/28/spacewatch-you-wait-ages-for-a-rocket-launch-then-",
        (
            "2019/dec/26/scientists-attempt-to-recreate-overview-effect-from"
            "-earth"
        ),
        (
            "2019/dec/15/exomars-race-against-time-to-launch-troubled-europe-"
            "mission-to-mars"
        ),
        (
            "2019/nov/06/cosmic-cats-nuclear-interstellar-messages-"
            "extraterrestrial-intelligence"
        ),
        (
            "2019/nov/14/spacewatch-boeing-proposes-direct-flights-moon-2024"
            "-nasa"
        ),
        "2019/nov/24/mars-robot-will-send-samples-to-earth",
        "2019/nov/06/daniel-lobb-obituary",
        (
            "2019/dec/09/european-space-agency-to-launch-clearspace-1-space-"
            "debris-collector-in-2025"
        ),
        (
            "2020/feb/27/biggest-cosmic-explosion-ever-detected-makes-huge-"
            "dent-in-space"
        ),
        (
            "2020/feb/06/christina-koch-returns-to-earth-after-record-"
            "breaking-space-mission"
        ),
        (
            "2020/jan/01/international-space-station-astronauts-play-with-"
            "fire-for-research"
        ),
        "2020/jan/05/space-race-moon-mars-asteroids-commercial-launches",
        "2019/oct/08/nobel-prizes-have-a-point-parking-space",
        "2019/oct/31/spacewatch-nasa-tests-new-imaging-technology-in-space",
        "blog/2020/feb/06/can-we-predict-the-weather-in-space",
        "2019/sep/08/salyut-1-beat-skylab-in-space-station-race",
        (
            "2020/feb/13/not-just-a-space-potato-nasa-unveils-astonishing-"
            "details-of-most-distant-object-ever-visited-arrokoth"
        ),
    ]
    guardian_urls = [
        f"[{k+1}]({base_url}/{url})" for k, url in enumerate(guardian_urls)
    ]
    # for url in guardian_urls:
    #     print(url)
    acceptable_urls = ", ".join(guardian_urls)
    acceptable_urls = "**ACCEPTABLE URLS TO USE:** " + acceptable_urls
    return acceptable_urls


def create_header_image_html():
    d = {
        (
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
            "master/man/figures/lter_penguins.png"
        ): [
            "40%",
            "Allison Horst",
        ],
        (
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
            "master/man/figures/culmen_depth.png"
        ): ["30%", "Allison Horst"],
    }
    image_str = " ".join(
        [f"<img src={k} width={v[0]} />" for k, v in d.items()]
    )
    pictures_credits = "Pictures by " + ", ".join(
        [f"[{v[1]}]({k})" for k, v in d.items()]
    )
    # print(pictures_credits)
    return [image_str, pictures_credits]


def create_description_header_html():
    desc = (
        "Predicts the topic of Space news articles from the Science section "
        "of the [Guardian News Media](https://www.theguardian.com/science/"
        "space) website from November 6, 2019 to February 28, 2020."
    )
    return desc

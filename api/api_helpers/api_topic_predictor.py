#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from api_helpers.api_processing_helpers import process_text


def add_datepart(df):
    df[["year", "month", "day"]] = df["url"].str.extract(
        r"/(\d{4})/([a-z]{3})/(\d{2})/"
    )
    d = {"jan": 1, "feb": 2, "nov": 11, "dec": 12, "sep": 9, "oct": 10}
    df["month"] = df["month"].map(d).astype(int)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["month"] = df["month"].map({v: k.title() for k, v in d.items()})
    df["weekday"] = df["date"].dt.day_name()
    df["week_of_month"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
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
    base_cols = [
        "url",
        "date",
        "year",
        "week_of_month",
        "weekday",
        "month",
        "text",
    ]
    df_topics_new = df_topics[
        base_cols + ["topic_num", "topic"] + list(range(n_top_words))
    ].merge(
        df_named_topics.reset_index()[["topic_num", "best"]], on="topic_num"
    )[
        [
            "url",
            "date",
            "year",
            "week_of_month",
            "weekday",
            "month",
            "text",
            "topic_num",
            "topic",
            "best",
        ]
    ]
    df_topic_word_factors = (
        topic_words.groupby(topic_words.index)
        .apply(lambda x: x.iloc[0].nlargest(n_top_words))
        .reset_index()
        .rename(
            columns={"level_0": "topic_num", "level_1": "term", 0: "weight"}
        )
    )
    return [df_topics_new, df_topic_word_factors]


def generate_response(
    df_new,
    pipe,
    n_top_words,
    n_topics_wanted,
    df_named_topics,
    df_residuals_new,
):
    df_predicted_new_topics, df_topic_word_factors = make_predictions(
        add_datepart(df_new),
        pipe,
        n_top_words,
        n_topics_wanted,
        df_named_topics,
    )
    df_predicted_new_topics = df_predicted_new_topics.drop(
        columns=["topic"], axis=1
    ).rename(columns={"best": "topic"})
    dtype_dict = {"topic_num": int}
    df_topic_word_factors = df_topic_word_factors.astype(dtype_dict).rename(
        columns={"weight": "term_weight"}
    )
    cols_wanted = [
        "url",
        "date",
        "year",
        "week_of_month",
        "weekday",
        "month",
        "topic_num",
        "topic",
        "term",
        "term_weight",
        "entity",
        "entity_count",
    ]
    weights_series = df_topic_word_factors.groupby("topic_num")[
        ["term", "term_weight"]
    ].agg(list)
    ents_series = df_residuals_new.groupby("topic")[
        ["entity", "entity_count"]
    ].agg(list)
    df_response = df_predicted_new_topics.merge(
        weights_series.reset_index(), on="topic_num"
    ).merge(ents_series.reset_index(), on="topic")[cols_wanted]
    # print(df_response.head(10))
    # print(df_residuals_new.head(10))
    df_residuals_new_reshaped = (
        df_residuals_new.groupby("topic")[
            ["resid_min", "resid_max", "resid_perc25", "resid_perc75"]
        ]
        .first()
        .reset_index()
    )
    df_response = df_response.merge(df_residuals_new_reshaped, on="topic")
    # print(list(df_response))
    # print(df_response.shape)
    # print(df_response.iloc[:5, :10])
    # print(df_response.iloc[:5, 10:])
    # for k, row in df_predicted_new_topics.iterrows():
    #     print(f"Row={k}, URL={row['url']}, Topic={row['best']}\n")
    return df_response.to_dict("records")

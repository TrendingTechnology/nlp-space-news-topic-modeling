#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import peewee as pw
from api_helpers.api_processing_helpers import process_text

# Training NMF residual stastics
# - from 8_*.ipynb (after last histogram of section 8)
training_residuals_stats = {
    "mean": 0.8877953271898069,
    "25%": 0.8537005649638325,
    "50%": 0.9072421433186082,
    "max": 0.9948739031036548,
}
# Acceptable percent diff between training and inference residual statistics
training_residual_stat_margin = 5


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


def get_residual(pipe, new_texts):
    A = pipe.named_steps["vectorizer"].transform(new_texts)
    W = pipe.named_steps["nmf"].components_
    H = pipe.named_steps["nmf"].transform(A)
    f"A={A.toarray().shape}, W={W.shape}, H={H.shape}"
    r = np.zeros(A.shape[0])
    for row in range(A.shape[0]):
        r[row] = np.linalg.norm(A[row, :] - H[row, :].dot(W), "fro")
    return r


def get_documentwise_residuals(
    pipe, df, training_res_stats, training_residual_stat_margin
):
    # Get residuals
    unseen_residuals_series = pd.Series(
        get_residual(pipe, df["processed_text"].tolist()), index=df.index
    )
    # if len(df) == 1:
    #     unseen_residuals_valid_series = pd.Series(
    #         [False] * len(df), index=df.index
    #     )
    #     print(
    #         f"Single unseen observation detected. "
    #         "Set placeholder value for residual."
    #     )
    #     return [unseen_residuals_valid_series, unseen_residuals_series]
    # Get statistics on residuals
    training_res_stats = pd.DataFrame.from_dict(
        training_residuals_stats, orient="index"
    ).rename(columns={0: "training"})
    unseen_res_stats = (
        unseen_residuals_series.describe()
        .loc[["mean", "25%", "50%", "max"]]
        .rename("unseen")
        .to_frame()
    )
    df_residual_stats = training_res_stats.merge(
        unseen_res_stats, left_index=True, right_index=True
    )
    df_residual_stats["diff_pct"] = percentage_change(
        df_residual_stats["training"], df_residual_stats["unseen"]
    )
    # Sanity check
    try:
        assert (
            df_residual_stats["diff_pct"] <= training_residual_stat_margin
        ).all()
        unseen_residuals_valid_series = pd.Series(
            [True] * len(df), index=df.index
        )
        print(
            "Unseen data residual distribution within bounds compared to "
            "training."
        )
    except AssertionError as e:
        unseen_residuals_valid_series = pd.Series(
            [False] * len(df), index=df.index
        )
        print(
            f"{str(e)} - Unseen data residuals OOB compared to training. "
            "Set placeholder value."
        )
    return [unseen_residuals_valid_series, unseen_residuals_series]


def percentage_change(col1, col2):
    return ((col2 - col1) / col1) * 100


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
        pd.DataFrame(doc_topic)
        .idxmax(axis=1)
        .rename("topic_num")
        .to_frame()
        .assign(url=df["url"].tolist())
    )
    assert df_temp["url"].tolist() == df["url"].tolist()

    merged_topic = df_temp.merge(topic_df, on="topic_num", how="left").assign(
        url=df["url"].tolist()
    )
    df_topics = df.merge(merged_topic, on="url", how="left").astype(
        {"topic_num": int}
    )
    assert pd.to_datetime(df_topics["date"]).is_monotonic
    assert df_topics["url"].tolist() == df["url"].tolist()
    assert df["url"].equals(df_topics["url"])
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
        df_named_topics.reset_index()[["topic_num", "best"]],
        on="topic_num",
        how="left",
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
    (
        df_topics_new["valid_resid_distribution"],
        df_topics_new["resid"],
    ) = get_documentwise_residuals(
        pipe, df, training_residuals_stats, training_residual_stat_margin
    )

    df_topic_word_factors = (
        topic_words.groupby(topic_words.index)
        .apply(lambda x: x.iloc[0].nlargest(n_top_words))
        .reset_index()
        .rename(
            columns={"level_0": "topic_num", "level_1": "term", 0: "weight"}
        )
    )
    return [df_topics_new, df_topic_word_factors]


def check_list(row, row_keys=["term", "term_weight"], key_types=[str, float]):
    obs = f"Observation {int(row.name)+1}"
    for row_value, key_type, row_key in zip(
        [row[row_key] for row_key in row_keys], key_types, row_keys
    ):
        # print(row_value)
        try:
            assert type(row_value) == list
            assert len(row_value) <= 10
            assert all(type(item) == key_type for item in row_value)
            if row_key in ["term_weight", "entity_count"]:
                # print(row_key, row_value)
                if row_value:
                    assert row[f"valid_{row_key}"]
                    assert all(
                        earlier >= later
                        for earlier, later in zip(row_value, row_value[1:])
                    )
                    print(f"{obs} - Passed DESC order check for {row_key}")
                else:
                    assert not row_value
                    assert not row[f"valid_{row_key}"]
                    print(f"{obs} - Passed empty list check for {row_key}")
        except AssertionError as e:
            row_value = []
            row[f"valid_{row_key}"] = False
            print(
                f"{str(e)} - {obs} - Failed list check. "
                f"Set placeholder values for {row_key}"
            )


def check_residuals(row, dfres):
    obs = f"Observation {int(row.name)+1}"
    resid_cols = ["resid_min", "resid_perc25", "resid_perc75", "resid_max"]
    for row_key in resid_cols:
        # print(row_key)
        try:
            assert type(row[row_key]) == float
            assert row[row_key] > 0 and row[row_key] < 1
            mask = dfres["topic"] == row["topic"]
            assert row[row_key] == dfres[mask][row_key].iloc[0]
            if row_key in [resid_cols[0]]:
                assert row["valid_residual"]
            print(f"{obs} - Passed check for {row_key}")
        except AssertionError as e:
            try:
                assert row[row_key] == -999.999
                assert not row["valid_residual"]
                print(
                    f"{str(e)} - {obs} - Failed check for {row_key}. "
                    "Detected placeholder value."
                )
            except AssertionError as e:
                row[row_key] = -999.999
                row["valid_residual"] = False
                print(
                    f"{str(e)} - {obs} - Failed check for {row_key}. "
                    "Set placeholder value."
                )

    if all(row[resid_cols] != -999.999):
        try:
            first, second = itertools.tee(row[resid_cols].tolist())
            next(second)
            assert all(
                later >= earlier for earlier, later in zip(first, second)
            )
            print(f"{obs} - Passed ASC order check for residual cols")
        except AssertionError as e:
            try:
                assert all(row[resid_cols] == -999.999)
                assert not row["valid_residual"]
                print(
                    (
                        f"{str(e)} - {obs} - Failed ASC order check for "
                        "residuals. Detected placeholder."
                    )
                )
            except AssertionError as e:
                row[resid_cols] = -999.999
                row["valid_residual"] = False
                print(
                    (
                        f"{str(e)} - {obs} - Failed ASC order check for "
                        "residuals. Set placeholder."
                    )
                )


def generate_response(
    df_new,
    pipe,
    n_top_words,
    n_topics_wanted,
    df_named_topics,
    df_residuals_new,
    # Prediction,
    # DB,
    n_days=1,
    df_erroneous=pd.DataFrame(),
):
    # pipe comes from nlp_pipe.joblib
    # df_named_topics comes from nlp_topic_names.csv
    df_new = add_datepart(df_new)
    data_n_days = (df_new["date"].max() - df_new["date"].min()).days
    # print(n_days, data_n_days, type(n_days), type(data_n_days))
    df_new = df_new.sort_values(by=["date"]).reset_index(drop=True)
    df_predicted_new_topics, df_topic_word_factors = make_predictions(
        df_new,
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
        "resid",
        "valid_resid_distribution",
        "term",
        "term_weight",
        "entity",
        "entity_count",
    ]
    weights_series = df_topic_word_factors.groupby("topic_num")[
        ["term", "term_weight"]
    ].agg(list)
    df_response = df_predicted_new_topics.merge(
        weights_series.reset_index(), on="topic_num", how="left"
    )

    ents_series = pd.Series()
    df_residuals_new_reshaped = pd.DataFrame()
    if data_n_days == n_days:
        # df_residuals_new comes from nlp_topic_residuals.csv
        ents_series = df_residuals_new.groupby("topic")[
            ["entity", "entity_count"]
        ].agg(list)
        df_response = df_response.merge(
            ents_series.reset_index(), on="topic", how="left"
        )[cols_wanted]
    assert df_new["url"].equals(df_response["url"])
    assert df_new["url"].tolist() == df_response["url"].tolist()
    # print(df_response.head(10))
    # print(df_residuals_new.head(10))

    validity_cols = [
        "valid_residual",
        "valid_term_weight",
        "valid_entity_count",
    ]
    df_response[validity_cols] = True
    if data_n_days == n_days:
        df_residuals_new_reshaped = (
            df_residuals_new.groupby("topic")[
                ["resid_min", "resid_max", "resid_perc25", "resid_perc75"]
            ]
            .first()
            .reset_index()
        )
        df_response = df_response.merge(
            df_residuals_new_reshaped, on="topic", how="left"
        )
    else:
        all_url_cols = [
            "entity",
            "entity_count",
        ]
        for col in all_url_cols:
            df_response[col] = [[]] * len(df_response)
        df_response["valid_entity_count"] = False
        df_response[
            [
                "resid_min",
                "resid_max",
                "resid_perc25",
                "resid_perc75",
            ]
        ] = -999.999
        df_response["valid_residual"] = False
    assert df_new["url"].equals(df_response["url"])
    assert df_new["url"].tolist() == df_response["url"].tolist()
    # print(list(df_response))
    # print(df_response.shape)
    # print(df_response.iloc[:5, :10])
    # print(df_response.iloc[:5, 10:])
    # for k, row in df_predicted_new_topics.iterrows():
    #     print(f"Row={k}, URL={row['url']}, Topic={row['best']}\n")

    # Perform sanity checks and, if failing, set placeholder values
    df_response.apply(lambda x: check_residuals(x, df_residuals_new), axis=1)
    term_weight_cols = ["term", "term_weight"]
    df_response.apply(
        lambda x: check_list(x, term_weight_cols, [str, float]), axis=1
    )
    entity_counts_cols = ["entity", "entity_count"]
    df_response.apply(
        lambda x: check_list(x, entity_counts_cols, [str, int]), axis=1
    )
    print(df_response.dtypes)

    for key in ["entity_count", "term_weight", "entity", "term"]:
        df_response[key] = [
            "[" + ", ".join(map(str, df_response_col)) + "]"
            for df_response_col in df_response[key]
        ]

    if not df_erroneous.empty:
        d_dummy = {
            "date": pd.date_range(
                start="1500-01-01", periods=len(df_erroneous), freq="D"
            ),
            "year": ["-999"] * len(df_erroneous),
            "week_of_month": [-999] * len(df_erroneous),
            "weekday": ["-999"] * len(df_erroneous),
            "month": ["-999"] * len(df_erroneous),
            "topic_num": [-999] * len(df_erroneous),
            "topic": ["News article too short"] * len(df_erroneous),
            "resid": [-999.999] * len(df_erroneous),
            "valid_resid_distribution": [False] * len(df_erroneous),
            "term": ["-999"] * len(df_erroneous),
            "term_weight": ["-999"] * len(df_erroneous),
            "entity": ["-999"] * len(df_erroneous),
            "entity_count": ["-999"] * len(df_erroneous),
            "valid_residual": [False] * len(df_erroneous),
            "valid_term_weight": [False] * len(df_erroneous),
            "valid_entity_count": [False] * len(df_erroneous),
            "resid_min": [-999.999] * len(df_erroneous),
            "resid_max": [-999.999] * len(df_erroneous),
            "resid_perc25": [-999.999] * len(df_erroneous),
            "resid_perc75": [-999.999] * len(df_erroneous),
        }
        df_dummy = pd.DataFrame.from_dict(d_dummy, orient="index")
        df_dummy.index = df_erroneous.index
        df_erroneous = df_erroneous[["url"]].merge(
            df_dummy, left_index=True, right_index=True
        )
        df_response = pd.concat([df_response, df_erroneous])

    response_dict = df_response.to_dict("records")

    cols_order = [
        "url",
        "date",
        "year",
        "week_of_month",
        "weekday",
        "month",
        "topic_num",
        "topic",
        "resid",
        "valid_resid_distribution",
        "term",
        "term_weight",
        "entity",
        "entity_count",
        "valid_residual",
        "valid_term_weight",
        "valid_entity_count",
        "resid_min",
        "resid_max",
        "resid_perc25",
        "resid_perc75",
    ]

    for _, r in enumerate(df_response[cols_order].to_dict("records")):
        r["url"] = r["url"].lower()
        r["date"] = r["date"].to_pydatetime()
        # print(k + 1, type(r["date"]), type(r["url"]))
    #     p = Prediction(**r)
    #     try:
    #         p.save()
    #         print(f"Observation {k+1} save to Database completed")
    #     except pw.IntegrityError:
    #         print(f"Observation {k+1} error when saving to database")
    #         DB.rollback()
    # print(f"List of tables in database = {DB.get_tables()}")
    return response_dict

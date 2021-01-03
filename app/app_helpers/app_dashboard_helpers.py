#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
from datetime import date, datetime
from functools import lru_cache

import pandas as pd
import panel as pn
from jinja2 import Template


@lru_cache()
def load_data(data_filepath):
    df = pd.read_hdf(data_filepath, "df")
    return df


@lru_cache()
def get_data(
    start_date, end_date, topic_name, data_filepath="data/dashboard_data.h5"
):
    df_new = load_data(data_filepath)
    # print(df_new[["date", "topic"]])
    start_date_str = datetime.strptime(str(start_date), "%Y-%m-%d")
    end_date_str = datetime.strptime(str(end_date), "%Y-%m-%d")
    gt_mask = df_new["date"] >= start_date_str
    lt_mask = df_new["date"] <= end_date_str
    df_new_filtered = df_new.loc[(gt_mask) & (lt_mask)]
    # print(df_new[["date", "topic"]])
    df_topic_filtered = df_new[df_new["topic"].isin([topic_name])]
    # print(df_topic_filtered.shape[0])
    # print(df_topic_filtered)
    return [df_new_filtered, df_topic_filtered]


def generate_tooltip(
    data_list=[["term", "Term", 14], ["term_weight", "Term Weight", 14]],
    imgs=[],
    tool_title_fontsize=16,
    tool_title="My Title",
    jinja2_templates_dir="templates",
    hover_template="hover.j2",
):
    jinja2_templates_filepath = os.path.join(
        jinja2_templates_dir, hover_template
    )
    with open(jinja2_templates_filepath) as f:
        hover_tool_str = Template(f.read())
    if imgs:
        return hover_tool_str.render(
            data_list=data_list,
            tool_title_fontsize=tool_title_fontsize,
            tool_title=tool_title,
        )
    else:
        return hover_tool_str.render(
            data_list=data_list,
            tool_title_fontsize=tool_title_fontsize,
            tool_title=tool_title,
            imgs=imgs,
        )


def create_dashboard_sidebar_text():
    nlp_wiki_link = "https://en.wikipedia.org/wiki/Natural_language_processing"
    guardian_wiki_link = "https://en.wikipedia.org/wiki/The_Guardian"
    usl_url = (
        "https://en.wikipedia.org/wiki/Machine_learning#Unsupervised_learning"
    )
    nmfurl = "https://en.wikipedia.org/wiki/Non-negative_matrix_factorization"
    tm_link = "https://en.wikipedia.org/wiki/Topic_model"
    dash_title = f"""
    <h1><a href='{nlp_wiki_link}'>NLP</a>
    <a href='{tm_link}'>Topic Predictor</a></h1>"""
    text = f"""{dash_title}
        This dashboard visualizes the learned topic of news articles from the
        Science section of the
        <a href='{guardian_wiki_link}'>Guardian Online publication</a></h1>
        using the search term <b>Space</b>. An
        <a href='{usl_url}'>unsupervised Machine Learning</a></h1>
        <a href='{nmfurl}'>Machine Learning</a></h1>
        model was trained on the article text of approximately 4,000 news
        articles from the result of such a search on the
        <a href='https://www.theguardian.com/science'>guardian.com</a></h1>
        website covering the years 1957-late 2019. The trained ML model was
        used to generate predictions on 48 news articles from November 2019 to
        the end of February 2020 that were not seen during training. This
        dashboard summarizes select characteristics of the trained model and
        the learned topics for all of these unseen news articles.

        <h3>Notes</h3>
        Of the 35 learned topics, news articles in topics 0 and 33 were not
        manually read. These two topics were not assigned a name and so are
        listed as <i>Topic 0</i> and <i>Topic 33</i> respectively. If the
        vertical axis text is cut off on the plots, then click the menu icon
        (three horizontal bars at the top left) twice. This dashboard must be
        viewed in landscape mode.
        """
    return text


def update_progress(
    start_date, end_date, min_date, max_date, date_range_progress_bar
):
    selected_days = (end_date - start_date).days
    max_days = (max_date - min_date).days
    date_range_progress_bar.value = int((selected_days / max_days) * 100)


def generate_sidebar_html():
    return pn.pane.HTML(
        create_dashboard_sidebar_text(),
        width=450,
        style={
            "background-color": "#F6F6F6",  # text background color
            "border": "2px solid lightgrey",
            "border-radius": "3px",  # >0px produces curved corners
            "padding": "5px",  # text-to-border whitespace
        },
    )


def perform_updates(
    start_date,
    end_date,
    daterange_progress_bar,
    topic_selected,
    data_filepath,
    term_weights_bar_chart,
    entity_counts_bar_chart,
    topic_freq_bar_chart,
    source,
    source_weights,
    source_ner,
    source_month,
    source_topics,
    beginning_date,
    ending_date,
    topic_selector,
):
    df_new_filtered, df_topic_filtered = get_data(
        start_date, end_date, topic_selected, data_filepath
    )
    update_progress(
        start_date,
        end_date,
        beginning_date,
        ending_date,
        daterange_progress_bar,
    )
    # print(df_new_filtered[["date", "topic"]])
    # print(df_new_filtered["date"].value_counts().sort_index().reset_index())
    df_weekdays = (
        df_new_filtered["weekday"]
        .value_counts()
        .sort_values()
        .reset_index()
        .rename(columns={"weekday": "count", "index": "weekday"})
    )
    df_month = (
        df_new_filtered["month"]
        .value_counts()
        .sort_values()
        .reset_index()
        .rename(columns={"month": "count", "index": "month"})
    )
    df_topics = (
        df_new_filtered["topic"]
        .value_counts()
        .sort_values()
        .reset_index()
        .rename(columns={"topic": "count", "index": "topic"})
    )
    # print(df_topic_filtered)
    d_terms = {
        "term_weight": df_topic_filtered.iloc[0]["term_weight"],
        "term": df_topic_filtered.iloc[0]["term"],
        "topic": [topic_selected] * len(df_topic_filtered.iloc[0]["term"]),
    }
    d_ner = {
        "entity_count": df_topic_filtered.iloc[0]["entity_count"],
        "entity": df_topic_filtered.iloc[0]["entity"],
        "topic": [topic_selected] * len(df_topic_filtered.iloc[0]["entity"]),
    }
    # print(d_terms)
    df_terms = pd.DataFrame.from_dict(d_terms, orient="index").T
    source_weights.data = df_terms
    term_weights_bar_chart.y_range.factors = df_terms["term"].tolist()[::-1]
    t_dates = pd.to_datetime(df_topic_filtered["date"]).dt.strftime("%Y-%m-%d")
    # print(t_dates)
    topic_date_str = (
        f"{topic_selected} ({t_dates.iloc[0]} - {t_dates.iloc[-1]})"
    )
    term_weights_bar_chart.title.text = topic_date_str

    df_ner = pd.DataFrame.from_dict(d_ner, orient="index").T
    source_ner.data = df_ner
    entity_counts_bar_chart.y_range.factors = df_ner["entity"].tolist()[::-1]
    ner_date_str = (
        f"Occurrences of Organizations "
        f"({t_dates.iloc[0]} - {t_dates.iloc[-1]})"
    )
    entity_counts_bar_chart.title.text = ner_date_str
    entity_counts_bar_chart.xaxis.axis_label = topic_selected
    source.data = df_weekdays
    source_month.data = df_month
    source_topics.data = df_topics
    topic_freq_bar_chart.y_range.factors = df_topics["topic"].tolist()

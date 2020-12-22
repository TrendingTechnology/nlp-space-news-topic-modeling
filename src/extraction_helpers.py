#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_top_n_most_freq_words(df, n=10, return_pct=False):
    # https://stackoverflow.com/a/56712583/4057186
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(df["text"].str.join(","))
    word_list = cv.get_feature_names()
    count_list = cv_fit.toarray().sum(axis=0)
    df_top_words = pd.DataFrame.from_dict(
        dict(zip(word_list, count_list)), orient="index"
    )
    if return_pct:
        df_top_words[0] = (df_top_words[0] / df_top_words[0].sum()) * 100
    df_top_words = df_top_words.nlargest(n, 0)
    return df_top_words.to_dict()

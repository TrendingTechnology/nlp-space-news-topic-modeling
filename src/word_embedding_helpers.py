#!/usr/bin/python3
# -*- coding: utf-8 -*-


import re
from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import NMF


def calculate_coherence(w2v_model: Word2Vec, term_rankings: list) -> float:
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations(term_rankings[topic_index], 2):
            pair_scores.append(w2v_model.wv.similarity(pair[0], pair[1]))
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)


def get_descriptor(
    all_terms: np.array, H: np.array, topic_index, top: int
) -> list:
    # reverse sort the values to sort the indices
    top_indices = np.argsort(H[topic_index, :])[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append(all_terms[term_index])
    return top_terms


class TokenGenerator:
    def __init__(self, documents, stopwords):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile(r"(?u)\b\w\w+\b")

    def __iter__(self):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall(doc):
                if tok in self.stopwords:
                    tokens.append("<stopword>")
                elif len(tok) >= 2:
                    tokens.append(tok)
            yield tokens


def fit_nmf_for_num_topics(
    start: int,
    limit: int,
    random_state: int,
    doc_term_matrix,
    max_iter: int = 700,
) -> list:
    topic_models = []
    for num_topics in range(start, limit + 1):
        print(f" > Applying NMF with {num_topics:0d} topics...", end="")
        model = NMF(
            n_components=num_topics,
            max_iter=max_iter,
            random_state=random_state,
        )
        model_transformed = model.fit_transform(doc_term_matrix)
        print("done")
        factors_dict = model.components_
        topic_models.append((num_topics, model_transformed, factors_dict))
    return topic_models


def compute_coherence_values_manually(
    topic_models_fitted: list,
    feature_names: np.array,
    n_top_words: int,
    word2vec_model_fitted: Word2Vec,
) -> list:
    k_values = []
    coherences = []
    for (k, _, H) in topic_models_fitted:
        # Get all of the topic descriptors - the term_rankings, based on
        # top 10 terms
        term_rankings = []
        for topic_index in range(k):
            term_rankings.append(
                get_descriptor(feature_names, H, topic_index, n_top_words)
            )
        # Now calculate the coherence based on our Word2vec model
        k_values.append(k)
        coherences.append(
            calculate_coherence(word2vec_model_fitted, term_rankings)
        )
        print(
            f" > Applied NMF for k={k:0d} and found coherence="
            f"{coherences[-1]:.4f}"
        )
    return coherences


def print_top_words(
    topic_models: list,
    num_topics: int,
    n_top_words: int,
    start: int,
    feature_names: list,
    doc_term_matrix,
    method: int = 2,
    random_state: int = 42,
) -> np.array:
    print(f"Top terms per topic, using random_state={random_state}:")
    if method == 1:
        # get the model that we generated earlier.
        W, H = topic_models[num_topics - start][1:]
        for topic_index in range(n_top_words):
            descriptor = get_descriptor(
                feature_names, H, topic_index, n_top_words
            )
            str_descriptor = " ".join(descriptor)
            print(f"Topic {topic_index+1:0d}: {str_descriptor}")
        docs_topics = W
    else:
        # https://scikit-learn.org/stable/auto_examples/applications/
        # plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-
        # applications-plot-topics-extraction-with-nmf-lda-py
        best_model = NMF(
            n_components=num_topics, max_iter=700, random_state=random_state
        ).fit(doc_term_matrix)
        end_n = -n_top_words - 1
        for topic_idx, topic in enumerate(best_model.components_):
            message = f"Topic {topic_idx:0d}: "
            message += " ".join(
                [feature_names[i] for i in topic.argsort()[:end_n:-1]]
            )
            print(message)
        docs_topics = best_model.transform(doc_term_matrix)
    return docs_topics


def get_docs_with_topics(
    docs_topics: np.array,
    num_topics: int,
    df_raw: pd.DataFrame,
    mapper_dict: Dict,
) -> pd.DataFrame:
    Vt = pd.DataFrame(
        docs_topics,
        index=None,
        columns=[f"component_{k+1}" for k in range(num_topics)],
    )
    Vt.index.name = "document"
    Vt = Vt.div(Vt.sum(axis=1), axis=0)
    Vt.insert(0, "text", df_raw["text"].to_numpy().tolist())
    df_with_topics = (
        df_raw.set_index("text")
        .merge(
            Vt.reset_index().set_index("text"),
            how="inner",
            left_index=True,
            right_index=True,
        )
        .reset_index(drop=False)
    )
    # print(df_with_topics)
    mask = df_with_topics.columns[
        df_with_topics.columns.str.contains("component")
    ]
    df_with_topics["most_popular_topic"] = df_with_topics[mask].idxmax(axis=1)
    df_with_topics = (
        df_with_topics.sort_values(by=["year"], ascending=True)
        .replace({"most_popular_topic": mapper_dict})
        .rename(columns=mapper_dict)
    )
    return df_with_topics

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import re
from typing import Dict

import pandas as pd
import spacy
from gensim.models import CoherenceModel, Phrases, nmf
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def compute_coherence_values(
    corpus,
    id2word,
    texts,
    limit,
    start=2,
    step=3,
    random_state=42,
    chunk_size=500,
):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_dict : Dict of NMF topic models
    coherence_values : Coherence values corresponding to the NMF model with
    respective number of topics
    """
    coherence_values = []
    model_dict = {}
    for num_topics in range(start, limit + 1, step):
        print(f" > Applying NMR for k={num_topics:0d}...", end="")
        model = nmf.Nmf(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            chunksize=chunk_size,  # no. of docs to be used per training chunk
            passes=10,
            kappa=1.0,
            minimum_probability=0.01,
            w_max_iter=200,
            w_stop_condition=0.0001,
            h_max_iter=50,
            h_stop_condition=0.001,
            eval_every=10,
            normalize=True,
            random_state=random_state,
        )
        model_dict[num_topics] = model
        print("computing coherence score...", end="")
        coherence_model = CoherenceModel(
            model=model, texts=texts, dictionary=id2word, coherence="c_v"
        )
        model_coherence_value = coherence_model.get_coherence()
        print(f"found coherence={model_coherence_value:.4f}")
        coherence_values.append(model_coherence_value)
    return model_dict, coherence_values


def plot_coherence_scores(coherence_vals, start, stop, step, fig_size=(8, 4)):
    _, ax = plt.subplots(figsize=fig_size)
    x = range(start, stop + 1, step)
    ax.plot(x, coherence_vals)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    max_coherence = max(coherence_vals)
    max_coherence_num_topics = coherence_vals.index(max_coherence)
    best_k = start + max_coherence_num_topics
    ax.annotate(
        f"{best_k:0d} topics",
        xy=(best_k, max_coherence),
        xytext=(best_k, max_coherence),
        textcoords="offset points",
        fontsize=16,
        # arrowprops=dict(facecolor="black", shrink=0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc=(0.8, 0.9, 0.9), ec="b", lw=2),
    )
    ax.set_title(
        "Coherence Score versus no. of topics", loc="left", fontweight="bold"
    )


def get_bigrams_trigrams(
    data_words, min_count_of_words=5, phrase_score_threshold=100
):
    # Build the bigram and trigram models
    bigram = Phrases(
        data_words,
        min_count=min_count_of_words,
        threshold=phrase_score_threshold,
    )  # higher threshold fewer phrases.
    trigram = Phrases(bigram[data_words], threshold=phrase_score_threshold)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_model = Phraser(bigram)
    trigram_model = Phraser(trigram)
    return bigram_model, trigram_model


def format_topics_sentences(model, corpus, df_source, topic_mapper_dict):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for row in model[corpus]:
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each doc
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series(
                        [int(topic_num), round(prop_topic, 4), topic_keywords]
                    ),
                    ignore_index=True,
                )
            else:
                break
    sent_topics_df.columns = [
        "most_popular_topic",
        "Perc_Contribution",
        "Topic_Keywords",
    ]
    # Add original text to the end of the output
    sent_topics_df = (
        sent_topics_df.rename(columns={0: "Text"})
        .astype({"most_popular_topic": int})
        .replace({"most_popular_topic": topic_mapper_dict})
    )
    df_with_topics = sent_topics_df.merge(
        df_source, left_index=True, right_index=True, how="inner"
    )
    return df_with_topics


def sent_to_words(sentences: list, min_length: int = 2, max_length: int = 15):
    for sentence in sentences:
        yield (
            simple_preprocess(
                str(sentence),
                deacc=True,
                min_len=min_length,
                max_len=max_length,
            )
        )  # deacc=True removes punctuations\


def remove_stopwords(texts, manual_stop_words_list):
    return [
        [
            word
            for word in simple_preprocess(str(doc))
            if word not in manual_stop_words_list
        ]
        for doc in texts
    ]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    # Initialize spacy 'en' model, keeping only tagger component (for
    # efficiency)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def print_top_words_gensim(
    gensim_nmf_model: nmf.Nmf,
    mapper_dict: Dict,
    top_n_words: int = 10,
    random_state: int = 42,
) -> None:
    twords = {}
    print(f"Top terms per topic, using random_state={random_state}:")
    for topic, word in gensim_nmf_model.show_topics(
        num_topics=len(mapper_dict), num_words=top_n_words
    ):
        words_cleaned = re.sub("[^A-Za-z ]+", "", word)
        twords[topic] = words_cleaned
        print(f"Topic {topic}:", words_cleaned.replace("  ", " "))

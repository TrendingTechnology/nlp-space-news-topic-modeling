#!/usr/bin/python3
# -*- coding: utf-8 -*-


from gensim.models.coherencemodel import CoherenceModel
from gensim.models.nmf import Nmf


def get_nmf_coherence_scores(corpus, texts, num_topics, dictionary):
    nmf = Nmf(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        chunksize=2000,
        passes=5,
        kappa=0.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42,
    )
    # Run the coherence model to get the score
    cm = CoherenceModel(
        model=nmf,
        texts=texts,  # needed for coherence="c_v"
        corpus=None,  # not needed for coherence="c_v"
        dictionary=dictionary,
        coherence="c_v",
        topn=20,
        processes=-1,
    )
    return round(cm.get_coherence(), 5)

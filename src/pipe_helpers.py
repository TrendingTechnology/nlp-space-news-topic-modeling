#!/usr/bin/python3
# -*- coding: utf-8 -*-


import re
import string

import nltk
import wordninja
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.lang.en import English

parser = English()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def make_dict_replacements_in_text(rep_dict, text):
    """
    Use dictionary to make multiple replacements of substrings in a single
    string
    """
    rep = dict((re.escape(k), v) for k, v in rep_dict.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text


def clean_text(s):
    """Cleaning text (single string)"""
    for rs in [
        "\d+",
        "\[.*?\]",
        "\w*\d\w*",
        f"[{re.escape(string.punctuation)}]",
        "[‘’“”…]",
    ]:
        s = re.sub(r"{}".format(rs), "", s)
        s = re.sub("\?|\.|\!|\/|\;|\:", "", s)
        s = s.strip()
        s = s.replace("\n", " ")
        s = s.replace("\r", " ")
        s = s.lower()
        s = re.sub("  +", "", s)

        # replace twitter @mentions
        mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
        s = mentionFinder.sub("@MENTION", s)

        # replace HTML symbols
        s = s.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    rep_dict = {"facebook": "", "pinterest": "", "twitter": ""}
    s = make_dict_replacements_in_text(rep_dict, s)
    return s


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Perform element-wise cleaning on a list
    Usage
    -----
    my_list = ["my text", "yours@here", "some notes"]
    pipe = Pipeline(steps=[("cleaner", TextCleaner(split=True)), ...])
    pipe.fit_transform(my_list)
    """

    def __init__(self, split=False):
        self.split = split

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.split:
            X_tr = [
                re.sub(
                    "[0-9]+", "", " ".join(wordninja.split(clean_text(elem)))
                )
                for elem in X
            ]
        else:
            X_tr = [clean_text(elem) for elem in X]
        return X_tr


class StemTokenizer:
    """Perform stemming"""

    def __init__(self, stopwords, stype="snowball", clean=True, split=True):
        self.all_stop_words = stopwords
        self.clean = clean
        self.split = split
        self.stype = stype
        if self.stype == "snowball":
            self.snl = SnowballStemmer("english")
        elif self.stype == "porter":
            self.snl = PorterStemmer(mode="NLTK_EXTENSIONS")

    def __call__(self, doc):
        if self.clean and not self.split:
            parser_obj = re.sub("[0-9]+", "", clean_text(doc))
        elif self.split and not self.clean:
            parser_obj = re.sub("[0-9]+", "", " ".join(wordninja.split(doc)))
        else:
            parser_obj = re.sub(
                "[0-9]+", "", " ".join(wordninja.split(clean_text(doc)))
            )
        stems = [
            self.snl.stem(t)
            for t in word_tokenize(parser_obj)
            if t not in self.all_stop_words
        ]
        return stems


class NLTKLemmaTokenizer(object):
    """Perform NLTK lemmatization"""

    def __init__(self, stopwords, clean=True, split=True, get_word_pos=True):
        self.all_stop_words = stopwords
        self.clean = clean
        self.split = split
        self.get_word_pos = get_word_pos
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        if self.clean and not self.split:
            parser_obj = re.sub("[0-9]+", "", clean_text(doc))
        elif self.split and not self.clean:
            parser_obj = re.sub("[0-9]+", "", " ".join(wordninja.split(doc)))
        else:
            parser_obj = re.sub(
                "[0-9]+", "", " ".join(wordninja.split(clean_text(doc)))
            )
        if self.get_word_pos:
            lemmas = [
                self.wnl.lemmatize(t, get_wordnet_pos(t))
                for t in word_tokenize(parser_obj)
                if t not in self.all_stop_words
            ]
        else:
            lemmas = [
                self.wnl.lemmatize(t)
                for t in word_tokenize(parser_obj)
                if t not in self.all_stop_words
            ]
        return lemmas


class SpacyLemmaTokenizer(object):
    """Perform Spacy lemmatization"""

    def __init__(self, stopwords, clean=True, split=True):
        self.all_stop_words = stopwords
        self.clean = clean
        self.split = split

    def __call__(self, doc):
        if self.clean and not self.split:
            parser_obj = re.sub("[0-9]+", "", clean_text(doc))
        elif self.split and not self.clean:
            parser_obj = re.sub("[0-9]+", "", " ".join(wordninja.split(doc)))
        else:
            parser_obj = re.sub(
                "[0-9]+", "", " ".join(wordninja.split(clean_text(doc)))
            )
        lemma_list = [
            t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_
            for t in parser(parser_obj)
        ]
        lemmas = list(set(lemma_list) - self.all_stop_words)
        return lemmas

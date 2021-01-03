#!/usr/bin/python3
# -*- coding: utf-8 -*-


import re
import string

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Contraction mapping dictionary
# SOURCES
# - https://stackoverflow.com/a/19794953/4057186
# - https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}

# Build list of stop words
custom_stop_words = ["said", "say", "like"]
all_stop_words = ENGLISH_STOP_WORDS.union(custom_stop_words)


# Tokenizing words
# - https://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
def do_tokenize(text):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def expand_match(match):
    # Replace contraction using mapping dict
    # - https://stackoverflow.com/a/62541145/4057186
    return contractions[match.group(0)]


def expand_contractions(text):
    # Expand a single contraction
    # https://stackoverflow.com/a/62541145/4057186
    contractions_compiled = re.compile(f"({'|'.join(contractions.keys())})")
    return contractions_compiled.sub(expand_match, text)


def process_text(text):
    # Tokenize
    text = do_tokenize(text)
    # Change to lowercase
    text = [each.lower() for each in text]
    # Remove numbers
    text = [re.sub("[0-9]+", "", each) for each in text]
    # Expand all contractions
    text = [expand_contractions(each) for each in text]
    # Word Stemming
    # - https://www.nltk.org/howto/stem.html
    text = [SnowballStemmer("english").stem(each) for each in text]
    # Remove punctuation
    # - https://stackoverflow.com/a/266162/4057186
    text = [w for w in text if w not in list(set(string.punctuation))]
    # Remove stopwords
    text = [w for w in text if w not in all_stop_words]
    # Remove single characters
    text = [each for each in text if len(each) > 1]
    # Remove compound words
    # https://www.grammarly.com/blog/open-and-closed-compound-words/
    text = [each for each in text if " " not in each]
    return text

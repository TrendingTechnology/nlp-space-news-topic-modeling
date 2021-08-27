#!/bin/bash


python3 -m spacy init fill-config \
    models/base_config.cfg \
    models/config.cfg

python3 -m spacy train models/config.cfg \
    --output ./models \
    --paths.train ./data/processed/train.spacy \
    --paths.dev ./data/processed/valid.spacy

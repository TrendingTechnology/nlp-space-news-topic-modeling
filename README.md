# [NLP Space News Topic Modeling](#nlp-space-news-topic-modeling)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/nlp-space-news-topic-modeling/master?urlpath=lab) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nlp-space-news-topic-modeling/master/4_nlp_trials.ipynb) [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/nlp-space-news-topic-modeling) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) ![CI](https://github.com/edesz/nlp-space-news-topic-modeling/workflows/CI/badge.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)

## [Table of Contents](#table-of-contents)
1. [Project Idea](#project-idea)
   * [Project Overview](#project-overview)
   * [Motivation](#motivation)
2. [Data acquisition](#data-acquisition)
   * [Primary data source](#primary-data-source)
   * [Supplementary data sources](#supplementary-data-sources)
   * [Data file creation](#data-file-creation)
3. [Analysis](#analysis)
4. [Usage](#usage)
5. [Project Organization](#project-organization)

## [Project Idea](#project-idea)

### [Project Overview](#project-overview)
This project aims to learn topics published in Space news from two news publications - [New York Times](https://www.nytimes.com/section/science/space) (US) and [Guardian](https://www.theguardian.com/science) (UK).

## [Motivation](#motivation)
The model/tool would give an idea of what Space news topics matter to each publication over time. For example, a space mission led by the European Space Agency (ESA) might be more relevant/important to the Guardian than to the New York Times. The two news publications<sup>[1](#myfootnote1)</sup> list articles on their web pages and machine learning is deployed to learn topics from this news articles.

<a name="myfootnote1">1</a>: articles were also retrieved from the blog Space.com and from the Hubble Telescope news archive, but these data sources were not used in analysis

## [Data acquisition](#data-acquisition)
### [Primary data source](#primary-data-source)
Data is acquired by using the official API provided by each publication. They are listed below
- [Guardian](https://open-platform.theguardian.com/)
- [New York Times](https://developer.nytimes.com/)

### [Supplementary data sources](#supplementary-data-sources)
Data is also acquired from articles published by the Hubble Telescope and blog publication Space.com
- [Hubble News API](http://hubblesite.org/api/documentation)
- [Space.com](https://www.space.com/)

Although these articles were acquired, they were not used in analysis.

### [Data file creation](#data-file-creation)
1. Use `1_get_list_of_urls.ipynb`
   - programmatically retrieves urls from API or archive of publication
   - retrieves metadata such as date and time, section, sub-section, headline/abstract/short summary, etc.
2. Use `2_scrape_urls.ipynb`
   - scrapes news article text from publication url
3. Use `3_merge_scraped_and_filter.ipynb`
   - merge metadata (`1_get_list_of_urls.ipynb`) with scraped article text (`2_scrape_urls.ipynb`)

## [Analysis](#anlysis)
Analysis will be performed using an un-supervised learning model. Details are included in the various notebooks in the root directory and listed [below](#usage).

## [Usage](#usage)
1. Clone this repository
   ```bash
   $ git clone
   ```
2. Create Python virtual environment, install packages and launch interactive Python platform
   ```bash
   $ make build
   ```
3. Run notebooks in the following order
   - `3_merge_scraped_and_filter.ipynb` ([view online](https://nbviewer.jupyter.org/github/edesz/nlp-space-news-tpoic-modeling/executed-notebooks/blob/master/3_merge_scraped_and_filter-20200709-171637.ipynb))
     - merge multiple files of articles text data retrieved from news publications API or archive
     - filter out articles of less than 500 words
     - export to `*.csv` file for use in unsupervised machine learning models
   - `4_nlp_trials.ipynb` ([view online](https://nbviewer.jupyter.org/github/edesz/nlp-space-news-tpoic-modeling/executed-notebooks/blob/master/4_nlp_trials-20200709-171724.ipynb))
     - uses `scikit-learn`'s implementation of NLP to learn topics
       - deploys stop words from [`scikit-learn`](https://awhan.wordpress.com/2016/06/05/scikit-learn-nlp-list-english-stopwords/), Spacy ([1](https://stackoverflow.com/a/41172279/4057186), [2](https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936)) and [NLTK](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
       - (optionally) uses [Stemming](http://www.nltk.org/howto/stem.html) and [Lemmatization](https://www.geeksforgeeks.org/python-lemmatization-with-nltk/) from NLTK
   - `5_corex_nlp_trials.ipynb` ([view online](https://nbviewer.jupyter.org/github/edesz/nlp-space-news-tpoic-modeling/executed-notebooks/blob/master/5_corex_nlp_trials-20200709-210200.ipynb))
     - uses achored NLP with [CoreEx](https://github.com/gregversteeg/corex_topic) to learn topics
   - `6_gensim_nlp_trials` ([view online](https://nbviewer.jupyter.org/github/edesz/nlp-space-news-tpoic-modeling/executed-notebooks/blob/master/6_gensim_nlp_trials-20200709-171834.ipynb))
     - experiments in selecting number of topics using
       - coherence score from word embedding model with Gensim and `sklearn` implementation of TFIDF with NMF
       - coherence score from Gensim implementation of NMF without TFIDF vectorization
   - `7_clustering_trials.ipynb` ([view online](https://nbviewer.jupyter.org/github/edesz/nlp-space-news-tpoic-modeling/executed-notebooks/blob/master/7_clustering_trials-20200709-172944.ipynb))
     - experiments in [KMeans clustering](https://en.wikipedia.org/wiki/K-means_clustering) with and without [LSA](https://en.wikipedia.org/wiki/Latent_semantic_analysis) for controlling dimensionality ([1](https://files.ifi.uzh.ch/stiller/CLOSER%202014/CSEDU/CSEDU/Information%20Technologies%20Supporting%20Learning/Full%20Papers/CSEDU_2014_88_CR.pdf), [2](https://www.quora.com/Can-I-combine-LSI-and-K-means-for-text-document-clustering-Are-there-any-sources-to-learn-about-it))

## [Project Organization](#project-organization)

    ├── .pre-commit-config.yaml       <- configuration file for pre-commit hooks
    ├── .github
    │   ├── workflows
    │       └── integrate.yml         <- configuration file for Github Actions
    ├── LICENSE
    ├── environment.yml               <- configuration file to create environment to run project on Binder
    ├── Makefile                      <- Makefile with commands like `make lint` or `make build`
    ├── README.md                     <- The top-level README for developers using this project.
    ├── data
    │   ├── raw                       <- raw data retrieved from news publication
    |   └── processed                 <- merged and filtered data
    ├── executed-notebooks            <- Notebooks with output.
    │   ├── 3_merge_scraped_and_filter-20200709-171637.ipynb
    │   ├── 4_nlp_trials-20200709-171724.ipynb
    │   ├── 5_corex_nlp_trials-20200709-210200.ipynb
    │   ├── 6_gensim_nlp_trials-20200709-171834.ipynb
    │   ├── 7_clustering_trials-20200709-172944.ipynb
    |
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description
    │
    ├── requirements.txt              <- packages required to execute all Jupyter notebooks interactively (not from CI)
    ├── setup.py                      <- makes project pip installable (pip install -e .) so `src` can be imported
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes {{cookiecutter.module_name}} a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    └── tox.ini                       <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #<a target="_blank" href="https://asciinema.org/a/244658">cookiecutterdatascience</a></small></p>

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Run notebooks."""


import os
from datetime import datetime
from glob import glob
from typing import Dict, List

import pandas as pd
import papermill as pm

PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
raw_data_dir = os.path.join(PROJ_ROOT_DIR, "data", "raw")
processed_data_dir = os.path.join(PROJ_ROOT_DIR, "data", "processed")
output_notebook_dir = os.path.join(PROJ_ROOT_DIR, "executed_notebooks")
one_dict_nb_name = "1_get_list_of_urls.ipynb"
two_dict_nb_name = "2_scrape_urls.ipynb"
three_dict_nb_name = "3_merge_scraped_and_filter.ipynb"
eight_dict_nb_name = "8_gensim_coherence_nlp_trials_v2.ipynb"
publication_name_for_nbs_six_seven = "guardian"
publications = ["space", "guardian", "hubble", "nytimes"]
cloud_run = True
raw_data_filepaths = {}
for fname in publications:
    full_filename = f"{fname}_com" if fname == "space" else fname
    raw_data_filepaths[fname] = os.path.join(
        PROJ_ROOT_DIR, "data", "raw", f"{full_filename}_urls.csv"
    )

one_dict = {
    "data_dir": raw_data_dir,
    "urls": {
        "guardian": "https://content.guardianapis.com/search",
        "hubble": "http://hubblesite.org/api/v3/news?page=all",
        "space": "https://www.space.com/archive",
    },
    "guardian_from_date": "1950-01-01",
    "guardian_to_date": "2019-11-01",
    "guardian_section": "science",
    "guardian_query": "space",
    "guardian_start_page_num": 1,
    "guardian_num_pages_wanted": 49,
    "guardian_api": "<api-key-here>",
    "guardian_query_min_delay": 2,
    "guardian_query_max_delay": 4,
    "hubble_article_fields_available": ["name", "news_id", "url"],
    "space_com_years": list(range(1999, 2019 + 1)),
    "nytimes_api": "<api-key-here>",
    "nytimes_query": "space",
    "nytimes_begin_date": "19500101",
    "nytimes_end_date": "20191101",
    "nytimes_start_page_num": 0,
    "nytimes_num_pages_wanted": -1,
    "nytimes_newspaper_lang": "en",
    "list_of_urls_file": raw_data_filepaths,
}

raw_data_filepaths.update(
    {
        "nytimes": os.path.join(
            PROJ_ROOT_DIR, "data", "raw", "nytimes_urls__*.csv"
        )
    }
)
two_dict = {
    "data_dir": raw_data_dir,
    "min_delay_between_scraped": 0,
    "max_delay_between_scraped": 1,
    "list_of_urls_file": raw_data_filepaths,
}
if not cloud_run:
    urls = {
        k: pd.read_csv(two_dict["list_of_urls_file"][k]["url"].tolist())
        for k in ["guardian", "hubble", "space"]
    }
    urls = {}
    urls["nytimes"] = pd.concat(
        [pd.read_csv(f) for f in glob(raw_data_filepaths["nytimes"])],
        axis=0,
        ignore_index=True,
    )["web_url"].tolist()
    two_dict["urls"] = urls

three_dict = {
    "data_dir": raw_data_dir,
    "processed_data_dir": processed_data_dir,
    "az_storage_container_name": "myconedesx7",
    "cloud_data": True,
    "hubble_inputs": {"blobedesz23": "urls", "blobedesz22": "text"},
    "hubble_processed_filename": "hubble_processed.csv",
    "nytimes_inputs": {
        "blobedesz27": "urls_1950_1989",
        "blobedesz28": "urls_1990_1999",
        "blobedesz29": "urls_2000_2019",
        "blobedesz24": "text1",
        "blobedesz25": "text2",
        "blobedesz26": "text3",
    },
    "nytimes_processed_filename": "nytimes_processed.csv",
    "space_inputs": {
        "blobedesz35": "urls",
        "blobedesz30": "text1",
        "blobedesz31": "text2",
        "blobedesz32": "text3",
        "blobedesz33": "text4",
        "blobedesz34": "text5",
    },
    "space_processed_filename": "space_processed.csv",
    "guardian_inputs": {
        "blobedesz21": "urls",
        "blobedesz19": "text1",
        "blobedesz20": "text2",
    },
    "guardian_processed_filename": "guardian_processed.csv",
}

eight_dict = dict(
    cloud_data=True,
    topic_nums=list(range(10, 45 + 5, 5)),
    n_top_words=10,
    unwanted_guardian_cols=[
        "webTitle",
        "id",
        "sectionId",
        "sectionName",
        "type",
        "isHosted",
        "pillarId",
        "pillarName",
        "page",
        "document_type",
        "apiUrl",
        "publication",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "quarter",
    ],
    run_spacy_medium_model=False,
)


def papermill_run_notebook(
    nb_dict: Dict, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebook with papermill.
    Parameters
    ----------
    nb_dict : Dict
        nested dictionary of parameters needed to run a single notebook with
        key as notebook name and value as dictionary of parameters and values
    Usage
    -----
    > import os
    > papermill_run_notebook(
          nb_dict={
              os.path.join(os.getcwd(), '0_demo.ipynb'): {'mylist': [1,2,3]}
          }
      )
    """
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = os.path.basename(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_notebook_dir}/{output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook,
            output_path=f"{output_notebook_dir}/{output_nb}",
            parameters=nb_params,
        )


def run_notebooks(
    notebook_list: List, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebooks from CLI.
    Parameters
    ----------
    nb_dict : List
        list of notebooks to be executed
    Usage
    -----
    > import os
    > PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
    > one_dict_nb_name = "a.ipynb
    > one_dict = {"a": 1}
    > run_notebook(
          notebook_list=[
              {os.path.join(PROJ_ROOT_DIR, one_dict_nb_name): one_dict}
          ]
      )
    """
    for nb in notebook_list:
        papermill_run_notebook(
            nb_dict=nb, output_notebook_dir=output_notebook_dir
        )


if __name__ == "__main__":
    PROJ_ROOT_DIR = os.getcwd()
    notebook_list = [
        {os.path.join(PROJ_ROOT_DIR, nb_name): input_dict}
        for nb_name, input_dict in zip(
            [three_dict_nb_name, eight_dict_nb_name],
            [three_dict, eight_dict],
        )
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )

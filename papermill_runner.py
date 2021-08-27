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
three_dict_nb_name = "3_merge_scraped_and_filter.ipynb"
eight_dict_nb_name = "8_gensim_coherence_nlp_trials_v2.ipynb"
cloud_run = True

three_dict = {
    "data_dir": raw_data_dir,
    "processed_data_dir": processed_data_dir,
    "az_storage_container_name": "myconedesx7",
    "cloud_data": cloud_run,
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

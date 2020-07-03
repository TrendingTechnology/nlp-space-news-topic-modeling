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
four_dict_nb_name = "4_nlp_trials.ipynb"
five_dict_nb_name = "5_corex_nlp_trials.ipynb"
cloud_run = True
raw_data_filepaths = {}
for fname in ["space", "guardian", "hubble", "nytimes"]:
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
    "guardian_api": "4e521fbc-7b13-4e08-b9c1-9ab29181bee6",
    "guardian_query_min_delay": 2,
    "guardian_query_max_delay": 4,
    "hubble_article_fields_available": ["name", "news_id", "url"],
    "space_com_years": list(range(1999, 2019 + 1)),
    "nytimes_api": "3EGfZWOwdNRGiCaYQbhker5S32sWO2n7",
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
            PROJ_ROOT_DIR, "data", "raw", f"{full_filename}_urls__*.csv"
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
    "hubble_filename": "hubble_urls.csv",
    "hubble_text": "hubble.csv",
    "hubble_processed_filename": "hubble_processed.csv",
    "nytimes_filename": "nytimes_urls__*.csv",
    "nytimes_text_filenames": [
        "nytimes.csv",
        # # use below to scrape only certain articles' text at once
        # # and then combine all tries together
        # "nytimes_1.csv",
        # "nytimes_3.csv",
        # "nytimes_2.csv",
    ],
    "nytimes_processed_filename": "nytimes_processed.csv",
    "space_filename": "space_com_urls.csv",
    "space_text_filenames": [
        # # use below to scrape only certain articles' text at once
        # # and then combine all tries together
        "space.csv",
        # "space_1.csv",
        # "space_2.csv",
        # "space_3.csv",
        # "space_4.csv",
        # "space_5.csv",
    ],
    "space_processed_filename": "space_processed.csv",
    "guardian_filename": "guardian_urls.csv",
    "guardian_text_filenames": [
        # # use below to scrape only certain articles' text at once
        # # and then combine all tries together
        "guardian.csv",
        # "guardian_1.csv",
        # "guardian_2.csv"
    ],
    "guardian_processed_filename": "guardian_processed.csv",
}

four_dict = {
    "publication_name": "guardian",
    "manual_stop_words": ["nt", "ll", "ve"],
    "mapping_dict": {
        "nytimes": {
            "component_1": "Academia",
            "component_2": "Shuttle Missions and Crashes",
            "component_3": "Digital",
            "component_4": "Mars Exploration",
            "component_5": "Imaging Stars - Astronomy",
            "component_6": "Rocket Launches - Testing and Moon Landing",
            "component_7": "Dark Matter theories",
            "component_8": "Planetary Research",
            "component_9": "Space Funding Bodies",
            "component_10": "ISS - USA and Russian segments",
            "component_11": "Gravity and Black Holes - Hawking",
            "component_12": "Global Warming",
            "component_13": "Studying Comets and Meteors",
            "component_14": "Soviet Union Spy Satellites",
            "component_15": "Discover of Sub-Atomic particles",
        },
        "guardian": {
            "component_1": "Academia",
            "component_2": "ISS - USA and Russian segments",
            "component_3": "Mars Exploration",
            "component_4": "Imaging Stars - Astronomy",
            "component_5": "Studying Comets and Meteors",
            "component_6": "Discover of Sub-Atomic particles",
            "component_7": "Rocket Launches - Moon Landing",
            "component_8": "Shuttle Missions and Crashes",
            "component_9": "Saturn Research",
            "component_10": "Space Funding Bodies",
            "component_11": "Objects crashing into Earth",
            "component_12": "Gravity and Black Holes - Hawking",
            "component_13": "Rocket Launches - Testing",
            "component_14": "Planetary Research",
            "component_15": "Global Warming",
        },
    },
    "n_topics_wanted": 15,
    "number_of_words_per_topic_to_show": 10,
}
four_dict["data_dir_path"] = str(
    os.path.join(
        processed_data_dir, f"{four_dict['publication_name']}_processed.csv"
    )
)

five_dict = {
    "publication_name": "guardian",
    "manual_stop_words": ["nt", "ll", "ve"],
    "mapping_dict": {
        "nytimes": {
            "component_1": "Academia",
            "component_2": "Shuttle Missions (no Crashes)",
            "component_3": "Digital",
            "component_4": "Mars Exploration",
            "component_5": "Imaging Stars - Astronomy",
            "component_6": "Rocket Launches - Testing and Moon Landing",
            "component_7": "Dark Matter theories",
            "component_8": "Planetary Research",
            "component_9": "Space Funding Bodies",
            "component_10": "ISS - USA and Russian segments",
            "component_11": "Gravity and Black Holes - Hawking",
            "component_12": "Global Warming",
            "component_13": "Studying Comets and Meteors (by children)",
            "component_14": "Soviet Union Spy Satellites",
            "component_15": "Discover of Sub-Atomic particles",
        },
        "guardian": {
            "component_1": "Academia",
            "component_2": "ISS - USA and Russian segments",
            "component_3": "Mars Exploration",
            "component_4": "Imaging Stars - Astronomy",
            "component_5": "Studying Comets and Meteors",
            "component_6": "Discover of Sub-Atomic particles",
            "component_7": "Rocket Launches - Moon Landing",
            "component_8": "Shuttle Missions and Crashes",
            "component_9": "Saturn Research",
            "component_10": "Space Funding Bodies",
            "component_11": "Objects crashing into Earth",
            "component_12": "Gravity and Black Holes - Hawking",
            "component_13": "Rocket Launches - Testing",
            "component_14": "Pluto Research",
            "component_15": "Global Warming",
        },
    },
    "corex_anchors": {
        "nytimes": [
            ["research", "science", "university"],
            ["space", "shuttle", "mission", "launch", "astronaut"],
            ["computer", "disk", "software", "memory"],
            ["mars", "rover", "life"],
            ["stars", "galaxy", "telescope"],
            ["moon", "lunar", "rocket", "nasa", "spacex"],
            ["universe", "theory", "matter"],
            ["planet", "solar", "spacecraft", "asteroid"],
            ["science", "research", "budget", "education"],
            ["station", "space", "mir", "nasa"],
            ["black", "hole", "hawking", "gravity"],
            ["warming", "climate", "ice", "carbon"],
            ["comet", "meteor", "halley"],
            ["soviet", "satellite", "weapons"],
            ["particles", "quantum", "neutrino", "theory"],
        ],
        "guardian": [
            ["people", "science", "brain"],
            ["station", "space", "mir", "nasa"],
            ["mars", "rover", "life"],
            ["stars", "galaxy", "telescope", "astronomer"],
            ["comet", "meteor", "lander", "dust"],
            ["particles", "higgs", "collider", "matter"],
            ["moon", "lunar", "rocket", "nasa", "apollo"],
            ["space", "shuttle", "mission", "launch", "astronaut"],
            ["cassini", "titan", "saturn"],
            ["science", "research", "budget", "education"],
            ["rock", "collision", "earth", "asteroid", "impact"],
            ["black", "hole", "universe", "gravity"],
            ["space", "launch", "rocket", "nasa", "spacex"],
            ["pluto", "horizons", "dwarf"],
            ["warming", "climate", "ice", "carbon"],
        ],
    },
    "corex_anchor_strength": 4,
    "number_of_words_per_topic_to_show": 10,
}
five_dict["data_dir_path"] = str(
    os.path.join(
        processed_data_dir, f"{five_dict['publication_name']}_processed.csv"
    )
)
five_dict["n_topics_wanted"] = len(
    f"{five_dict['corex_anchors'][five_dict['publication_name']]}"
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
            [
                # one_dict_nb_name,
                # two_dict_nb_name,
                # three_dict_nb_name,
                four_dict_nb_name,
                five_dict_nb_name,
            ],
            [one_dict, two_dict, three_dict, four_dict, five_dict],
        )
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )

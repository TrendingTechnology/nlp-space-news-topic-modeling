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
six_dict_nb_name = "6_gensim_coherence_nlp_trials.ipynb"
seven_dict_nb_name = "7_clustering_trials.ipynb"
eight_dict_nb_name = "8_gensim_coherence_nlp_trials_v2.ipynb"
publication_name_for_nbs_six_seven = "guardian"
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

four_dict = {
    "publication_name": "guardian",
    "cloud_run": True,
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
    "cloud_run": True,
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
five_dict["data_dir_path"] = os.path.join(
    processed_data_dir, f"{five_dict['publication_name']}_processed.csv"
)
five_dict["n_topics_wanted"] = len(
    f"{five_dict['corex_anchors'][five_dict['publication_name']]}"
)

six_dict = {
    "publication_name": publication_name_for_nbs_six_seven,
    "data_dir_path": os.path.join(
        processed_data_dir,
        f"{publication_name_for_nbs_six_seven}_processed.csv",
    ),
    "cloud_run": True,
    "manual_stop_words": ["nt", "ll", "ve"],
    "gensim_tfidf_mapping_dict": {
        "guardian": {
            "component_1": "Gravity and Black holes - Hawking",
            "component_2": "Rocket Launches - Testing",
            "component_3": "Mars Exploration",
            "component_4": "Academia",
            "component_5": "Studying Comets and Meteors",
            "component_6": "Discover of Sub-Atomic particles",
            "component_7": "Rocket Launches - Moon Landing",
            "component_8": "Shuttle Missions and Crashes",
            "component_9": "Global Warming",
            "component_10": "ISS - USA and Russian segments",
            "component_11": "Objects crashing into Earth",
            "component_12": "Space Funding Bodies",
            "component_13": "Imaging Stars - Astronomy",
            "component_14": "Saturn Research",
            "component_15": "Planetary Research",
        }
    },
    "gensim_non_tfidf_mapping_dict": {
        "guardian": {
            0: "Studying Comets and Meteors",
            1: "Rocket Launches - Testing",
            2: "Discover of Sub-Atomic particles",
            3: "Learning and Memory",
            4: "ISS",
            5: "Brain Research",
            6: "Academia",
            7: "Rocket Launches - Moon Landing",
            8: "Pseudo space-science and Humanity - Opinion",
            9: "Imaging Stars - Astronomy",
            10: "Planetary Research",
            11: "Global Warming and Climate Science",
            12: "Dark Matter Theories",
            13: "Space Funding Bodies",
            14: "Mars Exploration",
        }
    },
    "limit": 16,
    "start": 12,
    "step": 1,
    "n_top_words": 10,
    "random_state": 42,
}

seven_dict = {
    "publication_name": publication_name_for_nbs_six_seven,
    "data_dir_path": os.path.join(
        processed_data_dir,
        f"{publication_name_for_nbs_six_seven}_processed.csv",
    ),
    "cloud_run": True,
    "manual_stop_words": ["nt", "ll", "ve"],
    "mapping_dict": {
        "guardian": {
            0: "Gravity and Black Holes - Hawking",
            1: "Shuttle Missions and Crashes",
            2: "Global Warming",
            3: "Academia 2",
            4: "Studying Comets and Meteors",
            5: "Rocket Launches - Testing",
            6: "Discover of Sub-Atomic particles",
            7: "Academia 1",
            8: "Planetary Research",
            9: "Imaging Stars - Astronomy",
            10: "Objects crashing into Earth",
            11: "Rocket Launches - Moon Landing",
            12: "Sky Watching",
            13: "ISS - USA and Russian segment",
            14: "Mars Exploration",
        }
    },
    "mapping_dict_lsa": {
        "guardian": {
            0: "Academia 1",
            1: "Space Funding Bodies",
            2: "Studying Comets and Meteors",
            3: "Academia 2",
            4: "ISS - USA and Russian segment",
            5: "Shuttle Missions and Crashes",
            6: "Mars Exploration",
            7: "Planetary Research",
            8: "Imaging Stars - Astronomy",
            9: "Discover of Sub-Atomic particles",
            10: "Sky Watching",
            11: "Rocket Launches - Moon Landing",
            12: "Gravity and Black Holes - Hawking",
            13: "Rocket Launches - Testing",
            14: "Objects crashing into Earth",
        }
    },
    "minibatch": False,
    "kmeans_random_state": 42,
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
            [
                # one_dict_nb_name,
                # two_dict_nb_name,
                three_dict_nb_name,
                four_dict_nb_name,
                # five_dict_nb_name,
                six_dict_nb_name,
                seven_dict_nb_name,
                eight_dict_nb_name,
            ],
            [
                # one_dict,
                # two_dict,
                three_dict,
                four_dict,
                # five_dict,
                six_dict,
                seven_dict,
                eight_dict,
            ],
        )
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )

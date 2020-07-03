# -*- coding: utf-8 -*-

from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import papermill as pm
from invoke import Collection, task
from invoke.context import Context

PROJ_ROOT_DIR = Path.cwd()  # type: Path
data_folder = "data"  # type: str


nb_path = PROJ_ROOT_DIR
one_dict_nb_path = str(nb_path / "1_get_list_of_urls.ipynb")
one_dict = {
    one_dict_nb_path: {
        "data_dir": str(Path().cwd() / "data" / "raw"),
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
    }
}
one_data_dir_path = Path(str(one_dict[one_dict_nb_path]["data_dir"]))
one_dict[one_dict_nb_path]["list_of_urls_file"] = {
    "space": one_data_dir_path / "space_com_urls.csv",
    "guardian": one_data_dir_path / "guardian_urls.csv",
    "hubble": one_data_dir_path / "hubble_urls.csv",
    "nytimes": one_data_dir_path / "nytimes_urls.csv",
}
one_dict[one_dict_nb_path]["query_params"] = {
    "guardian": {
        "section": one_dict[one_dict_nb_path]["guardian_section"],
        "from-date": one_dict[one_dict_nb_path]["guardian_from_date"],
        "to-date": one_dict[one_dict_nb_path]["guardian_to_date"],
        "order-by": "oldest",
        "page-size": 100,
        "q": one_dict[one_dict_nb_path]["guardian_query"],
        "api-key": one_dict[one_dict_nb_path]["guardian_api"],
    },
    "hubble": {},
}
two_dict_nb_path = str(nb_path / "2_scrape_urls.ipynb")
two_dict = {
    two_dict_nb_path: {
        "data_dir": str(Path().cwd() / "data" / "raw"),
        "min_delay_between_scraped": 0,
        "max_delay_between_scraped": 1,
    }
}  # type: Dict[str, Any]
hubble_file = "hubble_urls.csv"
two_data_dir_path = Path(str(two_dict[two_dict_nb_path]["data_dir"]))
two_dict[two_dict_nb_path]["list_of_urls_file"] = {
    "space": two_data_dir_path / "space_com_urls.csv",
    "guardian": two_data_dir_path / "guardian_urls.csv",
    "hubble": two_data_dir_path / hubble_file,
    "nytimes": two_data_dir_path / "nytimes_urls__*.csv",
}
nytimes_list_of_urls = str(
    two_dict[two_dict_nb_path]["list_of_urls_file"]["nytimes"]
)
two_dict[two_dict_nb_path]["urls"] = {
    "guardian": pd.read_csv(
        two_dict[two_dict_nb_path]["list_of_urls_file"]["guardian"]
    )["webUrl"].tolist(),
    "hubble": pd.read_csv(
        two_dict[two_dict_nb_path]["list_of_urls_file"]["hubble"]
    )["url"].tolist(),
    "space": pd.read_csv(
        two_dict[two_dict_nb_path]["list_of_urls_file"]["space"]
    )["url"].tolist(),
    "nytimes": pd.concat(
        [pd.read_csv(f) for f in glob(nytimes_list_of_urls)],
        axis=0,
        ignore_index=True,
    )["web_url"].tolist(),
}
three_dict_nb_path = str(nb_path / "3_merge_scraped_and_filter.ipynb")
three_dict = {
    three_dict_nb_path: {
        "data_dir": str(Path().cwd() / "data" / "raw"),
        "processed_data_dir": str(Path().cwd() / "data" / "processed"),
        "hubble_filename": "hubble_urls.csv",
        "hubble_text": "hubble.csv",
        "hubble_processed_filename": "hubble_processed.csv",
        "nytimes_filename": "nytimes_urls__*.csv",
        "nytimes_text_filenames": [
            "nytimes.csv",
            # # use below if you scrape only certain articles' text at once
            # # and then want to combine all tries together
            # "nytimes_1.csv",
            # "nytimes_3.csv",
            # "nytimes_2.csv",
        ],
        "nytimes_processed_filename": "nytimes_processed.csv",
        "space_filename": "space_com_urls.csv",
        "space_text_filenames": [
            # # use below if you scrape only certain articles' text at once
            # # and then want to combine all tries together
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
            # # use below if you scrape only certain articles' text at once
            # # and then want to combine all tries together
            "guardian.csv",
            # "guardian_1.csv",
            # "guardian_2.csv"
        ],
        "guardian_processed_filename": "guardian_processed.csv",
    }
}
four_dict_nb_path = str(nb_path / "4_nlp_trials.ipynb")
four_dict = {
    four_dict_nb_path: {
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
}
publication_name = four_dict[four_dict_nb_path]["publication_name"]
four_dict[four_dict_nb_path]["data_dir_path"] = str(
    Path().cwd() / "data" / "processed" / f"{publication_name}_processed.csv"
)
five_dict_nb_path = str(nb_path / "5_corex_nlp_trials.ipynb")
five_dict = {
    five_dict_nb_path: {
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
}  # type: Dict[str, Any]
publication_name = five_dict[five_dict_nb_path]["publication_name"]
five_dict[five_dict_nb_path]["data_dir_path"] = str(
    Path().cwd() / "data" / "processed" / f"{publication_name}_processed.csv"
)
five_dict[five_dict_nb_path]["n_topics_wanted"] = len(
    f"{five_dict[five_dict_nb_path]['corex_anchors'][publication_name]}"
)


def run(command, hide=False):
    """Execute a command with Invoke."""
    ctx = Context()
    r = ctx.run(command, echo=True, pty=False, hide=hide)
    return r


def print_message(f_objs: Union[Dict, List], fext: str = ".ipynb") -> None:
    """Prints a message listing files processed"""
    f_objs_pr = (
        [str(p) for p in list(f_objs.keys())]
        if isinstance(f_objs, dict)
        else [str(p) for p in f_objs]
    )  # type: List[str]
    f_paths = f"{fext}\n".join(f_objs_pr)  # type: str
    print(f"\nProcessed files:\n{f_paths}\n")


@task
def nbconvert_run_notebook(ctx, nb_list=None):
    """Execute notebook with nbconvert"""
    for notebook in nb_list:
        # print(notebook)
        now = datetime.now().strftime("%Y%m%d-%H%M%S")  # type: datetime
        output_nb = str(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )  # type: str
        cmd = (
            "jupyter nbconvert --to notebook "
            "--ExecutePreprocessor.kernel_name=python3 "
            "--ExecutePreprocessor.timeout=600 "
            f"--execute {notebook} "
            f"--output {output_nb}"
        )  # type: str
        # print(cmd)
        run(cmd)


@task
def papermill_run_notebook(ctx, nb_dict=None):
    """Execute notebook with papermill"""
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")  # type: datetime
        output_nb = str(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )  # type: str
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook, output_path=output_nb, parameters=nb_params
        )


@task()
def train_deploy_model(ctx):
    """
    Execute notebooks to train model
    Usage
    -----
    invoke train-model
    """
    for nb in [one_dict, two_dict, three_dict, four_dict]:
        papermill_run_notebook(ctx, nb_dict=nb)
    nbconvert_run_notebook(ctx, nb_list=list(five_dict.keys()))


ns = Collection()
ns.add_task(train_deploy_model, name="train-model")

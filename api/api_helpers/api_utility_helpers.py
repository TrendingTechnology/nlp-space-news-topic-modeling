#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
from itertools import zip_longest

from jinja2 import Template
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def get_image_urls_dict(api_image_specs_filepath):
    panes = {}
    # print(api_image_specs_filepath)
    with open(api_image_specs_filepath) as f:
        # yf = f.read()
        ab = yaml.load(f)
    for _, v in ab.items():
        # print(k, v["height"])
        panes[v["url"]] = {"height": v["height"], "width": v["width"]}
    return panes


def grouper(iterable, n):
    # assumes iterable can be evenly divided into chunks of size n
    args = [iter(iterable)] * n
    return zip_longest(*args)


def list_to_chunked_lists(
    iterable_of_keys, dict_of_dicts, api_image_spec_filepath, row_size=5
):
    ub = list(grouper(iterable_of_keys, row_size))
    outer_list = []
    for e in ub:
        inner_list = []
        for k in e:
            d = {
                "url": k,
                "h": dict_of_dicts[k]["height"],
                "w": get_image_urls_dict(api_image_spec_filepath)[k]["width"],
            }
            inner_list.append(d)
        # print(inner_list)
        # print()
        outer_list.append(inner_list)
    return outer_list


def generate_linked_image_grid(
    image_list,
    jinja2_templates_dir="configs",
    image_grid_template="linked_image_grid.j2",
):
    jinja2_templates_filepath = os.path.join(
        jinja2_templates_dir, image_grid_template
    )
    with open(jinja2_templates_filepath) as f:
        html_str = (
            Template(f.read())
            .render(image_list=image_list)
            .replace("\n\n", "\n")
        )
    # print(html_str)
    return html_str


def create_acceptable_urls_markdown(unseen_urls_filepath):
    with open(unseen_urls_filepath, "r") as f:
        guardian_urls = [x.lstrip("- ").rstrip("\n") for x in f if "-" in x]
    guardian_urls = [f"[{k+1}]({url})" for k, url in enumerate(guardian_urls)]
    # for url in guardian_urls:
    #     print(url)
    acceptable_urls = ", ".join(guardian_urls)
    acceptable_urls = "**ACCEPTABLE URLS TO USE:** " + acceptable_urls
    return acceptable_urls


def create_description_header_html():
    desc = (
        "Predicts the topic of Space news articles from the Science section "
        "of the [Guardian News Media](https://www.theguardian.com/science/"
        "space) website from November 2, 2019 to February 28, 2020."
    )
    return desc


def get_description_html(
    jinja2_templates_dir,
    jinja2_image_template,
    unseen_urls_filepath,
    api_image_specs_filepath,
    num_images_per_row=6,
):
    # print(api_image_specs_filepath)
    image_list = list_to_chunked_lists(
        list(get_image_urls_dict(api_image_specs_filepath).keys()),
        get_image_urls_dict(api_image_specs_filepath),
        api_image_specs_filepath,
        num_images_per_row,
    )
    image_grid_html = generate_linked_image_grid(
        image_list, jinja2_templates_dir, jinja2_image_template
    )
    description = f"""
    {create_description_header_html()}

    {image_grid_html}
    Photos from [nasa.gov](https://www.nasa.gov/).

    {create_acceptable_urls_markdown(unseen_urls_filepath)}
    """
    return description.replace("    ", "")


def generate_return_docstring():
    return_docstring = """
        Returns
        -------
        topic, text and other predicted/metadata per article : Dict[str: str]
        - `url`
          - news article url
        - `date`
          - news article publication date
        - `year`
          - year of publication
        - `week_of_month`
          - week of month in which article was published
        - `weekday`
          - day of week on which article was published
        - `month`
          - month of year in which article was published
        - `text`
          - retrieved text of news article
        - `topic_num`
          - arbitrarily assigned topic internal number
        - `topic`
          - predicted name of topic
        - `term` *
          - list of top 10 NLP tokens, found during training
        - `term_weight` *
          - list of TFIDF term weights, found during training
        - `entity` *, **
          - list of up to 10 most frequently occurring organizations detected
        - `entity_count`*
          - list of respective number of occurrences of organizations detected
        - `resid_min` *, **
          - list of min. residual (Frobenius norm) b/w NMF approx./true data
        - `resid_max` *, **
          - list of max. residual between NMF approximation and true data
        - `resid_perc25` *, **
          - list of 25th percentile of residual b/w NMF approx. and true data
        - `resid_perc75` *, **
          - list of 75th percentile of residual b/w NMF approx. and true data
          ```
          {
              "url": "https://www.google.com",
              "date": "1900-01-01T00:00:00",
              "year": "1900",
              "week_of_month": 1,
              "weekday": "Monday",
              "month": "Jan",
              "text": "<full article text>",
              "topic_num": <internally assigned topic number from 1 to 35>,
              "topic": "<topic name>",
              "term": [
                  "abc",
                  "defg",
                  .
                  .
                  .
              ],
              "term_weight": [
                  0.100,
                  0.200,
                  .
                  .
                  .
              ],
              "entity": [
                  "abcd",
                  "efg",
                  "hijkl"
                  .
                  .
                  .
              ],
              "entity_count": [
                  2,
                  2,
                  1,
                  .
                  .
                  .
              ],
              "resid_min": 0.1,
              "resid_max": 0.2,
              "resid_perc25": 0.12
              "resid_perc75": 0.18
          }
          ```

          * list applies to each topic
          ** empty list if all acceptable unseen article urls are not used
        """
    return return_docstring

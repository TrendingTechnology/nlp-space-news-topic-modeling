#!/usr/bin/python3
# -*- coding: utf-8 -*-


import markdown
import numpy as np
import pandas as pd


def get_topic_descriptions(topic_desc_filepath="api_webpage_topics.md"):
    f = open(topic_desc_filepath, "r")
    htmlmarkdown = markdown.markdown(f.read())
    topic_descs = htmlmarkdown.split("<p>")
    topic_lists = []
    for topic_desc in topic_descs:
        cleaned_topic_desc = topic_desc.strip("</p>")
        topic_texts = cleaned_topic_desc.split("\n- ")
        topic_descs_list = [
            a.strip("</p>\n") for k, a in enumerate(topic_texts)
        ]
        topic_lists.append(topic_descs_list)

    topic_dict = {sl[0]: sl[1] for sl in topic_lists[1:]}
    # print(topic_dict)
    return topic_dict


def web_topics_dict():
    d = {
        "Mars mission updates": [
            2,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-globe",
                "divider_color": "8B0000",
                "url": (
                    "https://mars.nasa.gov/system/resources/detail_files/"
                    "6453_mars-globe-valles-marineris-enhanced-full2.jpg"
                ),
                "counter": 1,
            },
        ],
        "News Reports about Philae lander on Rosetta": [
            5,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-robot",
                "divider_color": "008000",
                "url": (
                    "https://solarsystem.nasa.gov/system/content_pages/"
                    "main_images/1348_philae.jpg"
                ),
                "counter": 2,
            },
        ],
        "SpaceX Rocket Testing Reports": [
            7,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-rocket",
                "divider_color": "0000CD",
                "url": (
                    "https://www.nasa.gov/sites/default/files/thumbnails/"
                    "image/crewdragon_iss_graphic.jpg"
                ),
                "counter": 3,
            },
        ],
        "Studying Earth-Threatening Asteroids": [
            8,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-meteor",
                "divider_color": "D2691E",
                "url": (
                    "https://www.nasa.gov/sites/default/files/thumbnails/"
                    "image/edu_asteroid_large.jpg"
                ),
                "counter": 4,
            },
        ],
        "On the Search for and Detection of Neutrinos": [
            26,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fab fa-elementor",
                "divider_color": "FF00FF",
                "url": (
                    "https://www.nasa.gov/sites/default/files/thumbnails/"
                    "image/suggested_video_thumbnail.png"
                ),
                "counter": 5,
            },
        ],
        "Discovery of Higgs Boson": [
            12,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-atom",
                "divider_color": "8A2BE2",
                "url": (
                    "https://scitechdaily.com/images/Standard-Model-of-"
                    "Particle-Physics.jpg"
                ),
                "counter": 6,
            },
        ],
        "Columbia shuttle crash": [
            1,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-space-shuttle",
                "divider_color": "FF1493",
                "url": (
                    "https://www.nasa.gov/sites/default/files/styles/"
                    "full_width/public/thumbnails/image/"
                    "columbia_rollout_for_sts-1_dec_29_1980.jpg?"
                    "itok=QRsLQjTy"
                ),
                "counter": 7,
            },
        ],
        "Cassini mission updates": [
            15,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-satellite",
                "divider_color": "800000",
                "url": (
                    "https://www.nasa.gov/images/content/60030main_cassini-"
                    "concept-browse.jpg"
                ),
                "counter": 8,
            },
        ],
        "Report on Detection of Gravitational Waves": [
            23,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fab fa-galactic-republic",
                "divider_color": "FF8C00",
                "url": (
                    "https://www.nasa.gov/sites/default/files/thumbnails/"
                    "image/blackhole20171003-16.jpg"
                ),
                "counter": 9,
            },
        ],
        "Beagle 2 Mission Updates": [
            28,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fas fa-microscope",
                "divider_color": "483D8B",
                "url": (
                    "https://solarsystem.nasa.gov/system/content_pages/"
                    "main_images/847_Artist_s_impression_of_Beagle_2_"
                    "lander_node_full_image_2.jpg"
                ),
                "counter": 10,
            },
        ],
        "Black Holes": [
            25,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fab fa-megaport",
                "divider_color": "FF4500",
                "url": (
                    "https://www.nasa.gov/sites/default/files/"
                    "cygx1_ill_0.jpg"
                ),
                "counter": 11,
            },
        ],
        "Search for E.T. life": [
            30,
            {
                "width": "80%",
                "height": "75%",
                "icon": "fab fa-wolf-pack-battalion",
                "divider_color": "191970",
                "url": (
                    "https://www.nasa.gov/sites/default/files/thumbnails/"
                    "image/hubble-pic.jpg"
                ),
                "counter": 12,
            },
        ],
    }
    ttd = get_topic_descriptions("api_helpers/api_webpage_topics.md")
    for k, v in d.items():
        v.insert(0, ttd[k])

    def df_to_html(
        df,
        card_table_caption=(
            "Summary Statistics for Topic Residuals during training"
        ),
    ):
        return (
            df.to_html(
                classes=["table", "table-bordered"],
                table_id="dataTable",
                index=False,
                border=0,
            )
            .replace(
                "<table",
                (
                    "<table data-toggle='table' width='100%' cellspacing='0' "
                    "data-header-style='headerStyle'"
                ),
            )
            .replace("tr style", "tr class='bg-info text-white' style")
            .replace(
                "<thead>",
                f"<caption>{card_table_caption}</caption><thead>",
            )
            .replace("<th>", "<th class='text-center'>")
        )

    # Get data
    # df = pd.DataFrame(
    #     np.random.rand(12, 5),
    #     columns=["min", "q25", "median", "q75", "max"],
    # )
    # df["count"] = list(range(1, 12 + 1))
    # df.index = list(d.keys())
    df = pd.read_csv("data/training_residuals_for_api_webpage.csv")
    df = df.set_index("best")

    # Set decimal places for HTML table display purposes
    table_formatter_dict = {c: "{:.3f}".format for c in list(df)[:-1]}
    table_formatter_dict[list(df)[-1]] = "{:.0f}".format
    df = df.apply(table_formatter_dict)
    # print(df)

    # Append table HTML to topic dict
    for k, v in d.items():
        df_html = df_to_html(df.loc[[k]])
        # print(k, df_html)
        v[2]["table_html"] = df_html
    return d

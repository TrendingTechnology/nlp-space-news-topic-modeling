#!/usr/bin/python3
# -*- coding: utf-8 -*-


def get_guardian_text_from_soup(soup) -> str:
    """Get Guardian text from soup object"""
    mydiv = soup.find("div", {"class": "article-body-commercial-selector"})
    # print(mydiv)
    if not mydiv:
        mydiv = soup.find("div", {"class": "content__article-body"})
    unwanted_tweets = mydiv.findAll(
        "figure", {"class": "element element-tweet"}
    )
    for unwanted in unwanted_tweets:
        unwanted.extract()
    unwanted_images = mydiv.findAll(
        "figure", {"class": "element element-embed"}
    )
    for unwanted in unwanted_images:
        unwanted.extract()
    unwanted_images2 = mydiv.findAll(
        "figure",
        {
            "class": (
                "element element-image "
                "img--landscape fig--narrow-caption fig--has-shares"
            )
        },
    )
    for unwanted in unwanted_images2:
        unwanted.extract()
    all_text = str(mydiv.text).replace("\n", "")
    art_text = all_text.split("Topics")[0]
    # print(art_text)
    return art_text

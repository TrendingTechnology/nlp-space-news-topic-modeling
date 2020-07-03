#!/usr/bin/env python3


def generate_space_archive_url(year, s_archive_base_url):
    """
    Generate list of space.com urls, using manually assembled
    lookup dict sp, based on archive at: https://www.space.com/archive
    """
    sp = {
        1999: [7],
        2000: [1, 4, 6],
        2001: [2],
        2002: [5, 7, 10],
        2003: [10, 11, 12],
        2004: [1, 2, 4, 5, 7, 8, 9, 10, 11, 12],
        2005: list(range(1, 12 + 1)),
        2006: list(range(1, 12 + 1)),
        2007: list(range(1, 12 + 1)),
        2008: list(range(1, 12 + 1)),
        2009: list(range(1, 12 + 1)),
        2010: list(range(1, 12 + 1)),
        2011: list(range(1, 12 + 1)),
        2012: list(range(1, 12 + 1)),
        2013: list(range(1, 12 + 1)),
        2014: list(range(1, 12 + 1)),
        2015: list(range(1, 12 + 1)),
        2016: list(range(1, 12 + 1)),
        2017: list(range(1, 12 + 1)),
        2018: list(range(1, 12 + 1)),
        2019: list(range(1, 11 + 1)),
    }
    return [f"{s_archive_base_url}/{year}/{month:02d}" for month in sp[year]]


def generate_nytimes_api_url(
    query,
    begin_date,
    end_date,
    api,
    fq_news_desk_contains="Science",
    fq_section_name_contains="Science",
):
    """Assemble NY Times API article url"""
    url = (
        "https://api.nytimes.com/svc/search/v2/articlesearch.json?"
        f'q="{query}"'
        f"&begin_date={begin_date}"
        f"&end_date={end_date}"
        "&page={0}"
        "&sort=oldest"
        '&document_type="article"'
        f'&fq=news_desk.contains:("{fq_news_desk_contains}")'
        f'AND section_name.contains:("{fq_section_name_contains}")'
        'AND type_of_material.contains:("news")'
        f"&api-key={api}"
    )
    # print(url)
    return url

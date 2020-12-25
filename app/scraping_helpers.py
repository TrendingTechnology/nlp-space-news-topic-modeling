#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os

from azure.storage.blob import BlobServiceClient


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


def download_az_file_blobs(blob_names_dict, az_container_name="myconedesx7"):
    conn_str = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT')};"
        f"AccountKey={os.getenv('AZURE_STORAGE_KEY')};"
        f"EndpointSuffix={os.getenv('ENDPOINT_SUFFIX')}"
    )
    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=conn_str
    )
    for az_blob_name, local_file_path in blob_names_dict.items():
        blob_client = blob_service_client.get_blob_client(
            container=az_container_name, blob=az_blob_name
        )
        with open(local_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
from io import StringIO

import pandas as pd


def load_data(
    cloud_data,
    data_dir,
    filename,
    text_filenames,
    guardian_inputs,
    blob_service_client,
    az_storage_container_name,
    unwanted_guardian_cols,
):
    if not cloud_data:
        df_listings = pd.read_csv(os.path.join(data_dir, filename))
        df_text = pd.concat(
            [pd.read_csv(os.path.join(data_dir, f)) for f in text_filenames]
        )
    else:
        guardian_dict = {}
        for az_blob_name, file_type in guardian_inputs.items():
            blob_client = blob_service_client.get_blob_client(
                container=az_storage_container_name, blob=az_blob_name
            )
            blobstring = blob_client.download_blob().content_as_text()
            guardian_dict[file_type] = pd.read_csv(StringIO(blobstring))
        df_text = pd.concat(
            [v for k, v in guardian_dict.items() if "text" in k]
        )
        df_listings = pd.concat(
            [v for k, v in guardian_dict.items() if k == "urls"]
        )

    df_listings.rename(
        columns={"webUrl": "url", "webPublicationDate": "publication_date"},
        inplace=True,
    )
    df_text.drop(["publication_date"], axis=1, inplace=True)

    df_text = df_text.set_index(["url"])
    df_listings = df_listings.set_index(["url"])

    df = df_text.merge(
        df_listings,
        left_index=True,
        right_index=True,
        how="inner",
    ).reset_index(drop=False)

    df = df[df["text"].str.len() > 500]
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.date
    df.drop(unwanted_guardian_cols, axis=1, inplace=True)
    return df

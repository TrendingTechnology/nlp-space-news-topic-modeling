#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Space News Article Topic Predictor Dashboard."""


import os

import app_helpers.app_data_retrieval_helpers as adrh

PROJ_ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(os.path.dirname(__file__))
data_filepath = os.path.join(PROJ_ROOT_DIR, "data", "dashboard_data.h5")

azure_blob_file_dict = {"blobedesz44": data_filepath}

# Data download from blob storage, if not found locally
if not os.path.exists(data_filepath):
    adrh.download_az_file_blobs(azure_blob_file_dict)

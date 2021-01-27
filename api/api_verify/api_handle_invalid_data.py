#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Verification of handling invalid requests."""


import json
import logging
import os
from urllib.parse import urljoin

import api_dummy_data_loader as adl
import pandas as pd
import requests

pd.set_option("display.max_columns", 5000)


def update_logging(k, r, single_obs=True):
    error_msg = ""
    logger = logging.getLogger(__name__)
    if r.status_code != 200:
        error_returned = json.loads(r.text)["detail"]
        # If multi-observation request fails
        if not single_obs:
            return error_returned
        # If single-observation request fails
        formatted_error_list = [
            msg["loc"][-1] + "=" + msg["msg"] for msg in error_returned
        ]
        error_msg = (
            f"Obs={k+1}, Status Code={r.status_code}, Msg=["
            f"{', '.join(formatted_error_list)+']'}"
        )
        logger.error(error_msg)
    else:
        # If multi-observation request is successful
        if not single_obs:
            response = json.loads(r.text)
            return response
        # If single-observation request is successful
        logger.info(
            f"Obs={k+1}, Status Code={r.status_code}, Msg=Successful Response"
        )
    return error_msg


if __name__ == "__main__":
    # Sensible CSV file texts, that don't break /uploadcv POST endpoint
    # - retrieving data
    PROJ_PAR_DIR = os.path.abspath(os.path.join(".", os.pardir))
    data_dir = os.path.join(PROJ_PAR_DIR, "data")

    # Load data
    real_filepath = os.path.join(data_dir, "guardian_3.csv")
    if not os.path.exists(real_filepath):
        df = adl.load_az_blob_data(
            az_storage_container_name="myconedesx7",
            az_blob_name="blobedesz42",
            n_rows=None,
            index_col=None,
        )
        df.to_csv(real_filepath, index=False)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # ENV_PORT = os.getenv("API_PORT", 8000)
    ENV_PORT = int(os.environ.get("API_PORT", 8000))
    HOST_URL = "0.0.0.0"
    # print(ENV_PORT)
    HOST_PORT = f"http://{HOST_URL}:{ENV_PORT}"

    # # Nonsense values to verify response of the /predict POST endpoint
    url = urljoin(f"{HOST_PORT}/api/v1/topics/", "predict").lower()
    headers = {"Content-Type": "application/json"}

    PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
    dummy_data_filepath = os.path.join(PROJ_ROOT_DIR, "dummy_url_inputs.json")

    # Generate dummy data
    single_obs_list, multi_obs_list = adl.get_dummy_invalid_url_data(
        dummy_data_filepath
    )
    # Verify multi-observation request
    r = requests.post(url, data=json.dumps(multi_obs_list), headers=headers)
    error_msg = update_logging(0, r, False)
    if error_msg:
        print(
            pd.DataFrame.from_records(error_msg).assign(
                status_code=r.status_code
            )
        )
    # Verify single-observation request
    for k, val in enumerate(single_obs_list):
        r = requests.post(url, data=json.dumps(val["data"]), headers=headers)
        error_msg = update_logging(k, r, True)
        assert r.status_code == val["status_code"]
        if r.status_code != 200:
            error_msg_str = (
                f"Obs={k+1}, Status Code={r.status_code}, Msg={val['msg']}"
            )
            assert error_msg_str == error_msg

    # # Nonsense CSV file texts, that break /uploadcv POST endpoint response
    url = urljoin(f"{HOST_PORT}/api/v1/topics/", "uploadcsv").lower()
    headers = {"accept": "application/json"}

    # Generate dummy data
    d_observation_csvs = adl.generate_dummy_invalid_csv_data()
    # Verification
    for dummy_data_filename, v in d_observation_csvs.items():
        file = adl.save_file_get_endpoint_dict(
            v["data"], dummy_data_filename, data_dir
        )
        r = requests.post(
            url,
            headers=headers,
            files=file,
        )
        assert v["status_code"] == r.status_code
        response_err_msg = json.loads(r.text)
        if response_err_msg["detail"]:
            assert v["error_msg"] in response_err_msg["detail"][0]["msg"]
            logger = logging.getLogger(__name__)
            logger.error(v["error_msg"])

    # Sensible CSV file texts, that don't break /uploadcv POST endpoint
    # Verification
    file_dict = {
        "data_filepath": (
            f"{real_filepath}",
            open(f"{real_filepath}", "rb"),
            "text/csv",
        ),
    }
    r = requests.post(
        url,
        headers=headers,
        files=file_dict,
    )
    try:
        assert r.status_code == 200
    except AssertionError as e:
        error_msg = f"{str(e)} - Did not get expected successful response!"
        logger = logging.getLogger(__name__)
        logger.critical(error_msg)
    else:
        logger = logging.getLogger(__name__)
        logger.info("Successful Response")

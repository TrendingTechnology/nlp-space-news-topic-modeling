#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime


def check_max_date_range(records, n_days):
    """
    Verify that NER and residual keys are present in response if articles
    cover full range of acceptable dates
    """
    start_date, end_date = [
        datetime.strptime(records[k]["date"], "%Y-%m-%dT%H:%M:%S")
        for k in [0, -1]
    ]
    data_n_days = (end_date - start_date).days
    addtl_keys_returned = [
        "entity",
        "entity_count",
        "resid_min",
        "resid_max",
        "resid_perc25",
        "resid_perc75",
    ]
    assert all(k in list(records[0].keys()) for k in addtl_keys_returned)
    for k in addtl_keys_returned:
        if k in ["entity", "entity_count", "term", "term_weight"]:
            if n_days != data_n_days:
                assert records[0][k] == "[]"
            else:
                assert records[0][k] != "[]"
        else:
            if n_days != data_n_days:
                assert records[0][k] == -999.999
            else:
                assert records[0][k] != -999.999

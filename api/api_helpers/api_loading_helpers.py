#!/usr/bin/python3
# -*- coding: utf-8 -*-


import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def handle_upload_file(
    upload_file: UploadFile, handler: Callable[[Path], None]
) -> None:
    tmp_path = save_upload_file_tmp(upload_file)
    try:
        df = handler(tmp_path)
    finally:
        tmp_path.unlink()
    return df

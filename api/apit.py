#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
from typing import Any, List, Mapping

import api_helpers.api_utility_helpers as lh
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
from pydantic import BaseModel

from app.router import api_router

PROJ_ROOT_DIR = os.getcwd()
jinja2_templates_dir = os.path.join(PROJ_ROOT_DIR, "templates")
unseen_urls_filepath = os.path.join(
    PROJ_ROOT_DIR, "api_helpers", "api_unseen_article_urls.yml"
)
api_image_specs_filepath = os.path.join(
    PROJ_ROOT_DIR, "api_helpers", "api_image_specs.yml"
)

description = lh.get_description_html(
    jinja2_templates_dir,
    "linked_image_grid.j2",
    unseen_urls_filepath,
    api_image_specs_filepath,
    6,
)
# print(description)

app = FastAPI(
    title="Space News Article Topic Predictor API",
    description=description,
    # docs_url="/",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates/")


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/form")
def form_post(request: Request):
    id = "1234"
    return templates.TemplateResponse(
        "index.html", context={"request": request, "id": id}
    )


app.include_router(api_router, prefix="/api/v1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

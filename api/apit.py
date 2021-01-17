#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os

import api_helpers.api_utility_helpers as lh
import api_helpers.api_webpage_banner_image_processing as apih
from api_helpers.api_webpage_topics import web_topics_dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.router import api_router

# ENV_PORT = os.getenv("PORT", 8000)
ENV_PORT = int(os.environ.get("PORT", 8000))
HOST_URL = "localhost"
# print(ENV_PORT)

banner_image_file_path = "static/assets/img/banner.jpg"
width_shrink_factor = 0.65
single_image_width = 1050
single_image_height = 1500
bt = 25
banner_urls = [
    (
        "https://www.nasa.gov/sites/default/files/thumbnails/image/"
        "parker_solar_probe_in_front_of_sun.jpg"
    ),
    (
        "https://www.nasa.gov/sites/default/files/thumbnails/image/nh-surface"
        ".jpg"
    ),
    (
        "https://www.nasa.gov/sites/default/files/thumbnails/image/"
        "the-moon-near-side.en_.jpg"
    ),
    "https://www.nasa.gov/images/content/136653main_s114e7221_high.jpg",
]

if not os.path.exists(banner_image_file_path):
    apih.create_banner_image(
        banner_urls,
        bt,
        width_shrink_factor,
        single_image_width,
        single_image_height,
        banner_image_file_path,
    )

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
all_args = {
    "topics": web_topics_dict(),
    "banner_urls": [[k, url] for k, url in enumerate(banner_urls)],
}
# print(description)

app = FastAPI(
    title="Space News Article Topic Predictor API",
    description=description,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates/")


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/webapp")
def webapp(request: Request):
    return templates.TemplateResponse(
        "index.html", context={"request": request, "values": all_args}
    )


app.include_router(api_router, prefix="/api/v1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

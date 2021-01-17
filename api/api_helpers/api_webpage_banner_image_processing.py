#!/usr/bin/python3
# -*- coding: utf-8 -*-


from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageOps


def crop_image(original, single_image_height, single_image_width):
    width, height = original.size  # Get dimensions
    # print(width, height)

    left_margin = (width / 2) - (single_image_width / 2)
    right_margin = (width / 2) + (single_image_width / 2)
    top_margin = (height / 2) - (single_image_height / 2)
    bottom_margin = (height / 2) + (single_image_height / 2)
    # print(left_margin, right_margin, top_margin, bottom_margin)

    cropped_example = original.crop(
        (left_margin, right_margin, top_margin, bottom_margin)
    )
    return cropped_example


def add_border(k, num_images, bt, cropped_example):
    if (k % 2) != 0:
        if k != num_images - 1:
            border = (0, bt, 0, bt)
        else:
            border = (0, bt, bt, bt)
    else:
        border = (bt, bt, bt, bt)
    bimg = ImageOps.expand(cropped_example, border=border, fill="white")
    return bimg


def process_image(
    k,
    image_url,
    num_images,
    single_image_width=300,
    single_image_height=400,
    border_thickness=50,
):
    response = requests.get(image_url)
    original = Image.open(BytesIO(response.content))
    # original.show()

    cropped_example = crop_image(
        original, single_image_height, single_image_width
    )
    bimg = add_border(k, num_images, border_thickness, cropped_example)
    return bimg


def combine_images(processed_images_dict):
    imgs_comb = np.hstack(
        [np.asarray(v) for _, v in processed_images_dict.items()]
    )
    imgs_comb = Image.fromarray(imgs_comb)
    return imgs_comb


def process_images(
    banner_urls, single_image_width, single_image_height, border_thickness
):
    processed_images = {}
    for k, url in enumerate(banner_urls):
        bimg = process_image(
            k,
            url,
            len(banner_urls),
            single_image_width,
            single_image_height,
            border_thickness,
        )
        # bimg.show()
        processed_images[k] = bimg
    # print(processed_images)
    return processed_images


def shrink_image(
    original, image_width_shrink_factor, single_image_width, num_images
):
    basewidth = int(
        (single_image_width * num_images) * (1 - image_width_shrink_factor)
    )
    wpercent = basewidth / float(original.size[0])
    hsize = int((float(original.size[1]) * float(wpercent)))
    shrunk_image = original.resize((basewidth, hsize), Image.ANTIALIAS)
    # shrunk_image.show()
    return shrunk_image


def create_banner_image(
    banner_urls,
    border_thickness=50,
    width_shrink_factor=0.65,
    single_image_width=1800,
    single_image_height=2400,
    banner_image_file_path="a.jpg",
):
    processed_images = process_images(
        banner_urls, single_image_width, single_image_height, border_thickness
    )
    combined_image = combine_images(processed_images)
    shrunk_image = shrink_image(
        combined_image,
        width_shrink_factor,
        single_image_width,
        len(banner_urls),
    )
    shrunk_image.save(banner_image_file_path)

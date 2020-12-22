#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""The calling script."""

import uvicorn

if __name__ == "__main__":
    # when running locally, use port 8000
    # when running from a Docker container with -p 8000:80, use port 80
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)

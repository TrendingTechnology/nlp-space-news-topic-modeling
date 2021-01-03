#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""The calling script."""

import uvicorn
from mangum import Mangum

from api import app

handler = Mangum(app)


if __name__ == "__main__":
    # when running locally, use port=8050
    # when running from a Docker container with -p 8000:8050, use port=8050
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8050,
        reload=True,  # during dev = True, in prod = False
    )

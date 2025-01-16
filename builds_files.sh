#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run collectstatic
python3 manage.py collectstatic --noinput

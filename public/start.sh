#!/bin/bash
export FLASK_APP=api/index.py
flask run --host=0.0.0.0 --port=$PORT

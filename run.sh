#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-reload

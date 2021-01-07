#!/bin/bash

find src/*.py | entr -cs "python src/main.py"


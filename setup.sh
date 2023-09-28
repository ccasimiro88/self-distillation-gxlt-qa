#!/bin/bash
# Setup python environment
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Extract corpora: MLQA, XQuAD, TyDiQA and SQuAD
tar -xvzf corpora.tar.gz
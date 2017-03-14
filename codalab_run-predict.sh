#! bin/bash

cl work main::cs224n-hahaha
cl upload code
cl upload data
cl upload train

cl run --name run-predict --request-docker-image sckoo/cs224n-squad:v4-0.12.1 :code :data :train dev.json:0x4870af2556994b0687a1927fcec66392  'python code/qa_answer.py --dev_path dev.json'

cl info --verbose run-predict


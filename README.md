# Amazon-LMRRC

The [Amazon Last-Mile Routing Research Challenge](https://routingchallenge.mit.edu/) focuses on the gap between theoretical route planning and real-life route execution that most optimization-based approaches are unable to bridge. This gap relies on several factors that are difficult to model explicitly, such as geography, infrastructure, and consumers drivers deliver to. The goal of the challenge is to bridge this gap exploiting a dataset of 6,112 historical routes operated by actual experienced drivers.

## How to download the data and use this code
You can use this code by simply cloning the repository. The dataset can be downloaded [here](), and once downloaded, the folder data must be located within the main directory of this repo. 

The three main Python script: model_build.py, model_apply.py, and model_score.py require a Linux environment to be run. This is done running the respective bash scrip in the Linux Terminal, e.g., ```bash model_build.sh```.

## How the code is structured
The code is structured in two parts:
1. Model Build
2. Model Apply

Read our [paper]() for more information about our methodolody. 
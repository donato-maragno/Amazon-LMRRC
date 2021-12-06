# Amazon-LMRRC

The [Amazon Last-Mile Routing Research Challenge](https://routingchallenge.mit.edu/) focuses on the gap between theoretical route planning and real-life route execution that most optimization-based approaches are unable to bridge. This gap relies on several factors that are difficult to model explicitly, such as geography, infrastructure, and consumers drivers deliver to. The goal of the challenge is to bridge this gap by exploiting a dataset of 6,112 historical routes operated by experienced drivers.

## How to download the data
You can use this code by simply cloning the repository. The dataset can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UFOG2H), and once downloaded, the folder ```data``` must be located within the main directory of this repo. 

## Methodology
The proposed methodology has a two-level hierarchical structure. First, a zone (geographical region with multiple stops) sequence is found solving a Traveling Salesman Problem (TSP), then within each zone an open TSP (OTSP) is solved. The predicted route is the concatenation of the OTSP solutions. The learning process takes place in the definition of a cost matrix used in the TSP for the zone sequence. The matrix is a weighted combination of travel times and transition probabilities. The latter is based on the number of times we see a transition from a zone to another one in the train data.

Within each zone, an OTSP is solved including forcing the first node to be the last of the previous zone, and the last node to be close to the center of the next zone. This gives a sense of direction in the resolution of the problem.

We refer to our [paper](https://arxiv.org/abs/2112.01937) for a detailed description of the methodology. 

## How the code is structured
The code is divided into three parts:
1. Model Build:  historical routes are used to define a probability matrix for the <em>zone</em> TSP.
2. Model Apply: the cost matrix is used to determine the zone sequence by formulating a TSP over the zones including the station, and when the zone sequence has been established the stop sequence within each zone is determined by means of a series of OTSPs.
3. Model Score: performance metrics are provided.

The three main Python scripts: model_build.py, model_apply.py, and model_score.py require a Linux environment to be run. 
They must be run sequentially using the bash command in the Linux Terminal, e.g., ```bash model_build.sh```.

## Citation
Our software can be cited as:
````
@misc{IILMR,
author = "Ghosh, Mayukh and Mahes, Roshan and Maragno, Donato",
title = "Incorporating Intuition in Last-Mile Routing",
year = 2021,
url = "https://github.com/donato-maragno/Amazon-LMRRC"
}
````

## Get in touch!
We welcome any questions or suggestions. Please submit an issue on GitHub, or reach us at m.ghosh@uva.nl, a.v.mahes@uva.nl, and d.maragno@uva.nl.

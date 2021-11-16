# Amazon-LMRRC

The [Amazon Last-Mile Routing Research Challenge](https://routingchallenge.mit.edu/) focuses on the gap between theoretical route planning and real-life route execution that most optimization-based approaches are unable to bridge. This gap relies on several factors that are difficult to model explicitly, such as geography, infrastructure, and consumers drivers deliver to. The goal of the challenge is to bridge this gap by exploiting a dataset of 6,112 historical routes operated by experienced drivers.

## How to download the data
You can use this code by simply cloning the repository. The dataset can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UFOG2H), and once downloaded, the folder data must be located within the main directory of this repo. 

## Methodology
The proposed methodology has a two-level hierarchical structure. First, a zone (geographical region with multiple stops) sequence is found solving a Travelling Salesman Problem (TSP), then within each zone an open TSP (OTSP) is solved. The predicted route is the concatenation of the OTSP solutions. The learning process takes place in the definition of a cost matrix used in the TSP for the zone sequence:


\[\label{eq3}
C_{ij} = 
\begin{cases}
\Omega_{ij} T_{ij} + (1 - \Omega_{ij})(1 - P_{ij}), &\quad i \neq j, \\
0, &\quad i = j.
\end{cases}\]


where $T_{ij}\in [0,1]$ is the normalized travel time from zone $i$ to zone $j$, and $P_{ij} \in [0,1]$ is the probability of going from zone $i$ to zone $j$. The probability matrix (P) is based on historical routes, and the more we see drivers going from zone $i$ to zone $j$, the higher $P_{ij}$ is. $\Omega$ is a weight which differs from station-to-zone, zone-to-zone, and zone-to-station.

Within each zone, an OTSP is solved including forcing the first node to be the last of the previous zone, and the last node to be close to the center of the next zone. This gives a sense of direction in the resolution of the problem.

We refer to our [paper]() for a detailed description of the methodology. 

## How the code is structured
The code is divided into three parts:
1. Model Build:  historical routes are used to define a probability matrix for the $zone$ TSP.
2. Model Apply: the cost matrix is used to determine the zone sequence by formulating a TSP over the zones including the station, and when the zone sequence has been established the stop sequence within each zone is determined by means of a series of OTSPs.
3. Model Score

The three main Python scripts: model_build.py, model_apply.py, and model_score.py require a Linux environment to be run. 
They must be run sequentially using the bash command in the Linux Terminal, e.g., ```bash model_build.sh```.
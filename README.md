# On the choice of graph neural network architectures
Clément Vignac, Guillermo Ortiz-Jiménez, Pascal Frossard, 
[On the choice of graph neural network architectures], ICASSP 2020

[TensorFlow] based benchmarking of graph neural networks on citation data.


This code is based on the [GNN-benchmark] library.

[On the choice of graph neural network architectures]: https://arxiv.org/abs/1911.05384 
[GNN-benchmark]: https://github.com/shchur/gnn-benchmark


[TensorFlow]: https://www.tensorflow.org

## Installation
Follow the instructions of [GNN-benchmark] for installing the library.
  
## Running experiments

Don't forget to turn MongoDB on before running the code.

### Experiment 1a.
In this experiment we compare the performance of [Graph Convolutional Networks]
and [Simple Graph Convolutions] on Cora, Pubmed and Citeseer when 50% of the
nodes are observed. Different numbers of random features are used.

``` bash
#!/usr/bin/env bash
python3 scripts/create_jobs.py -c config/experiment1a.conf.yaml --op search
python3 scripts/spawn_worker.py -c config/experiment1a.conf.yaml --gpu 0
python3 scripts/aggregate_results.py -c config/experiment1a.conf.yaml -o results/experiment1a/
```
To check that the code runs, you can reduce the number of splits in
 `config/experiment1a.conf.yaml`.

### Experiment 1b.
In this experiment the number of features is fixed (at 300), but the proportion
of observed nodes varies.

``` bash
#!/usr/bin/env bash
python3 scripts/create_jobs.py -c config/experiment1b.conf.yaml --op search
python3 scripts/spawn_worker.py -c config/experiment1b.conf.yaml --gpu 0


```


### Experiment 2

With 50% of observed nodes and 300 features, compare the performance of GCN,
SGC, APPNP as well as non-linear version of them (APPNP and SGC).

``` bash
#!/usr/bin/env bash
python3 scripts/create_jobs.py -c config/experiment2.conf.yaml --op fixed
python3 scripts/spawn_worker.py -c config/experiment2.conf.yaml --gpu 0
mkdir results/experiment2
python3 scripts/aggregate_results.py -c config/experiment2.conf.yaml -o results/experiment2/
```

[Graph Convolutional Networks]: https://arxiv.org/abs/1609.02907
[Simple Graph Convolutions]: https://arxiv.org/abs/1902.07153



## Cite
Please cite our paper as well as GNN-benchmark if you use this code:
```bibtex
@article{vignac2019choice,
  title={On the choice of graph neural network architectures},
  author={Vignac, Cl{\'e}ment and Ortiz-Jim{\'e}nez, Guillermo and Frossard, Pascal},
  journal={arXiv preprint arXiv:1911.05384},
  year={2019}
}
```


```bibtex
@article{shchur2018pitfalls,
  title={Pitfalls of Graph Neural Network Evaluation},
  author={Shchur, Oleksandr and Mumme, Maximilian and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
  journal={Relational Representation Learning Workshop, NeurIPS 2018},
  year={2018}
}
```

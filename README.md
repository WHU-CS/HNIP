# HNIP
This repository provides a reference implementation of HNIP proposed in "[Temporal Network Embedding with High-Order Nonlinear Information](https://aaai.org/Papers/AAAI/2020GB/AAAI-QiuZ.6746.pdf)", Zhenyu Qiu, Wenbin Hu, Jia Wu, Weiwei Liu, Bo Du and Xiaohua Jia, AAAI 2020

The HNIP algorithm learns a representations for nodes in a temporal graph.
Please check the paper for more details.

## Basic Usage
`
$ python main.py -c config/xx.ini
`

## Requirements
The implementation of HNIP is tested under Python 3.7, with the following packages installed:
* tensorflow-gpu==1.14.0
* numpy
* scipy


## Input
Your input graph data should be a **txt** file and be under **GraphData folder**

### Input format
The txt file should be adjlist with time stamp. In particular, the *i*-th line contains information 
about the *i*-th node and has the following structure:
`$$A:n_1,w_1,t_1;n_2,w_2,t_2;...;n_k,w_k,t_k$$`

where $A$ is the node ID, $n_1,...,n_k$ are the nodes adjacent to this node, 
$$w_1,...,w_k$$ are the weights of the links, and $$t_1,...,t_k$$ are the time stamps 
of the links. Note that the nodes are numbered starting from 0. Let $t_{max}$ be the 
time of the last link, $t_min$ be the time of the first link, and $t_c$ be the actual
time of the link $e_k$, then the $t_k = \frac{t_c - t_{min}}{t_{max}-t_{min}}$. 

### txt file  
```
0:1,1.0,0.34;2,1.0,0.33;
1:0,1.0,0.34;3,1.0,0.34;
...
```

## Output 
The output is the learned representation of the input network, all lines are node ID and *d*
dimensional representation:
```
0 0.0009352565 9.563565e-05 0.0013471842 ...
1 0.0009587407 9.6946955e-05 0.0013585389 ...
...
```

## Baselines
In our paper, we used the following methods for comparision:
* `DeepWalk` 'Deepwalk:online learning of social representations' [source](https://github.com/phanein/deepwalk.git)
* `Node2vec` ' node2vec: Scalable feature learning for networks' [source](https://github.com/aditya-grover/node2vec.git)
* `SDNE` 'Structural deep network embedding' [source](https://github.com/suanrong/SDNE.git)
* `CTDNE` 'Dynamic network embeddings: From random walks to temporal random walks' 
* `NetWalk` 'Netwalk: A flexible deep embedding approach for anomaly detection in dynamic networks' [source](https://github.com/chengw07/NetWalk.git)

*Note* that the `CTDNE` is not open-sourced, and we have implemented it based on the published paper. You 
can find the implementation in the *utils* folder.

## Citing
If you find HNIP useful in your research, we ask that you cite the following paper:
```
@inproceedings{DBLP:conf/aaai/QiuH00DJ20,
  author    = {Zhenyu Qiu and Wenbin Hu and Jia Wu and Weiwei Liu and Bo Du and Xiaohua Jia},
  title     = {Temporal Network Embedding with High-Order Nonlinear Information},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {5436--5443},
  year      = {2020}
}
```

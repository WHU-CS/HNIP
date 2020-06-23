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


## Input
Your input graph data should be a **txt** file and be under **GraphData folder**

### Input format
The txt file should be edgelist and the first line should be **N**, the number of nodes
and **E**, the number of links

### txt file  
`
191 1930
`

$a_1$


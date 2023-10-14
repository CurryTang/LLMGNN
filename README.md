# LLMGNN

Code for our paper [Label-free Node Classification on Graphs with Large Language Models (LLMS)](https://arxiv.org/abs/2310.04668). 

## Abstract
In recent years, there have been remarkable advancements in node classification achieved by Graph Neural Networks (GNNs). However, they necessitate abundant high-quality labels to ensure promising performance. In contrast, Large Language Models (LLMs) exhibit impressive zero-shot proficiency on text-attributed graphs. Yet, they face challenges in efficiently processing structural data and suffer from high inference costs. In light of these observations, this work introduces a label-free node classification on graphs with LLMs pipeline, LLM-GNN. It amalgamates the strengths of both GNNs and LLMs while mitigating their limitations. Specifically, LLMs are leveraged to annotate a small portion of nodes and then GNNs are trained on LLMs' annotations to make predictions for the remaining large portion of nodes. The implementation of LLM-GNN faces a unique challenge: how can we actively select nodes for LLMs to annotate and consequently enhance the GNN training? How can we leverage LLMs to obtain annotations of high quality, representativeness, and diversity, thereby enhancing GNN performance with less cost? To tackle this challenge, we develop an annotation quality heuristic and leverage the confidence scores derived from LLMs to advanced node selection. Comprehensive experimental results validate the effectiveness of LLM-GNN. In particular, LLM-GNN can achieve an accuracy of 74.9% on a vast-scale dataset \products with a cost less than 1 dollar.

![Pipeline demo](./imgs/pipeline.png)


## Environment Setups
```
conda env create -f environment.yml --name new_environment_name
```

Note: since the [faiss-gpu](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) has some conflicts because of the low version of GLIBC on the server, it's not included in this environment and I use it to generate centroids efficiently for large-scale graphs. I'll share the precomputed files later. 

## About the data

## How to use this repo and run the code

There are two main parts of our code
1. Annotators
2. GNN training

The pipeline works as follows: 
1. `get_dataset` in `data.py`: get the pt data file, use `llm.py` to generate annotations. The indexes selected by active learning and corresponding annotations will be returned. We use the cache to store all the output annotations. `-1` is a sentinel for null annotation. 
2. `main.py`: train the GNN models. For large-scale training, we do not use the batch version, but pre-compute all intermediate results. 

An example: 
```
python3 src/main.py --dataset products --model_name AdjGCN --data_format sbert --main_seed_num 1 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 940 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 50 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33
```



## Notes
I'll optimize the code structure when I have more time ⏳.

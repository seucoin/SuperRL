# SuperRL
Source code for ISWC2024 paper: Supervised Relational Learning with Selective Neighbor Entities for Few-Shot Knowledge Graph Completion(https://github.com/FKGC/SuperRL)

Few-shot Knowledge Graph (KG) completion is a focus of current research, where each task aims at querying unseen facts of a relation given limited reference triplets. However, existing works overlook two categories of neighbor entities relevant to few-shot relations, resulting in ineffective relational learning for few-shot KG completion. In this work, we propose a supervised relational learning model (SuperRL) with these crucial neighbor entities, where we design a cascaded embedding enhancement network to capture directly and indirectly relevant entities for few-shot relations and provide supervision signals by jointly performing dual contrastive learning and metric learning. Evaluation in link prediction on two public datasets shows that our method achieves new state-of-the-art results with different few-shot sizes.

# Requirements

```
python 3.6
Pytorch == 1.13.1
CUDA: 11.6
GPU: V100
```

# Datasets

We adopt Nell-One and Wiki-One datasets to evaluate our model, SuperRL.
The orginal datasets and pretrain embeddings are provided from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). 
For convenience, the datasets can be downloaded from [Nell data](https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz)
and [Wiki data](https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz). 
The pre-trained embeddings can be downloaded from [Nell embeddings](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
 and [Wiki embeddings](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing).
Note that all these files were provided by xiong and we just select what we need here. 
All the dataset files and the pre-trained TransE embeddings should be put into the directory ./data/NELL and ./data/Wiki, respectively.

# How to run
For optimal performance, please train SuperRL as follows:

#### Nell-One
3-shot
```
python main.py --fine_tune --num_layers 2 --lamda 0.06 --lr 8e-5 --few 3 --early_stop 10 --prefix SuperRL_c1n1-2_3_NELL
```

5-shot
```
python main.py --fine_tune --num_layers 2 --lamda 0.09 --lr 8e-5 --few 5 --early_stop 10 --prefix SuperRL_c1n1-2_5_NELL
```

#### Wiki-One
3-shot
```
python main.py --datapath "data/Wiki/" --num_layers 8 --lamda 0.06 --lr 2e-4 --few 3 --early_stop 10 --prefix new_final_c1n1-8_3_Wiki
```

5-shot
```
python main.py --datapath "data/Wiki/" --num_layers 8 --lamda 0.09 --lr 2e-4 --few 5 --early_stop 10 --prefix new_final_c1n1-8_5_Wiki
```

To test the trained models, please run as follows:

#### Nell-One

```
python main.py --test --num_layers 3 --lr 8e-5 --few 5 --prefix new_final_c1n1-3_5_NELL
```

#### Wiki-One

```
python main.py --test --datapath "data/Wiki/" --num_layers 6 --lr 2e-4 --prefix new_final_c1n1-6_5_Wiki
```

Here are explanations of some important args,

```bash
--data_path: "directory of dataset"
--few:       "the number of few in {few}-shot, as well as instance number in support set"
--num_layers:    "the number of enhancement layers"
--lamda:    "the trade-off parameter for dual contrastive loss"
--prefix:    "given name of current experiment"
--fine_tune  "whether to fine tune the pre_trained embeddings"
--device:    "the GPU number"
```

Normally, other args can be set to default values. See ``args.py`` for more details about argus if needed.


# FOREC: A Cross-Market Recommendation System
This repository provides the implementation of our CIKM 2021 paper titled as "[Cross-Market Product Recommendation](https://arxiv.org/pdf/2109.05929.pdf)". Please consider citing our paper if you find the code and [XMarket dataset](https://xmrec.github.io/) useful in your research. 

## Requirements:
We use conda for our experimentations. Please refer to the `requirements.txt` for the list of libraries we use for our implementation. After setting up your environment, you can simply run this command `pip install -r requirements.txt`. 

- python 3.7 
- pandas & numpy (pandas-1.3.3, numpy-1.21.2)
- torch==1.10.0
- [learn2learn](https://github.com/learnables/learn2learn)
- [pytrec_eval](https://github.com/cvangysel/pytrec_eval)

## DATA
The `DATA` folder in this repository contains the two target markets data (train ratings, validation run and qrel, and test run) that we conduct our evaluation on them and three source markets that you can use as desired to augment with the target markets. It is ultimately your choice on how to use the provided data (or any other additional data, if you will) to improve the recommendation performance on target markets. 


## Train the baseline models:
`train_baseline.py` is the script for training our simple GMF++ model that is taking one target market and zero to a few source markets for augmenting with the target market. We implemented our dataloader such that it loads all the data and samples equally from each market in the training phase. You can use ConcatDataset from `torch.utils.data` to concatenate your torch Datasets. 


Here is a sample train script using two source markets:

    python train_baseline.py --tgt_market t1 --src_markets s1-s2 --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name toytest --num_epoch 5 --cuda
    
Here is a sample train script using zero source market (only train on the target data):

    python train_baseline.py --tgt_market t1 --src_markets none --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name toytest --num_epoch 5 --cuda


After training your model, the scripts prints the directories of model and index checkpoints as well as the run files for the validation and test data as below. You can load the model for other usage and evaluate the validation run file. See the notebook `getting_started.ipynb` for a sample code on these. 

    Model is trained! and saved at:
    --model: checkpoints/t1_s1-s2_toytest.model
    --id_bank: checkpoints/t1_s1-s2_toytest.pickle
    Run output files:
    --validation: valid_t1_s1-s2_toytest.tsv
    --test: test_t1_s1-s2_toytest.tsv
    
You will need to upload the test run output file (.tsv file format) for both target markets to Codalab for our evaluation and leaderboard entry. This output file contains ranked items for each user with their score. Our final evaluation metric is based on nDCG@10 on both target markets.   



## FOREC
The general schema of our FOREC recommendation system is shown below. For a pair of markets, the middle part shows the market-agnostic model that we pre-train, and then fork and fine-tune for each market shown in the left and right. Note that FOREC is capable of working with any desired number of target markets. However, for simplicity, we only experiment with pairs of markets for the experiments. For further details, please refer to our paper. 


<p align="center">
  <img src="https://github.com/hamedrab/FOREC/blob/main/FOREC.png" width=80% height=80%>  
</p>







## Citation
If you use this dataset, please refer to our [CIKM’21 paper](https://arxiv.org/pdf/2109.05929.pdf):

```
@inproceedings{bonab2021crossmarket,
    author = {Bonab, Hamed and Aliannejadi, Mohammad and Vardasbi, Ali and Kanoulas, Evangelos and Allan, James},
    booktitle = {Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
    publisher = {ACM},
    title = {Cross-Market Product Recommendation},
    year = {2021}}
```

Please feel free to either open an issue or contacting me at bonab [AT] cs.umass.edu
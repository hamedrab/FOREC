"""
This script trains all GMF, MLP, and NMF baselines for a single market
Provides three options for the use of another source market:
  1. 'no_aug'  : only use the target market train data, hence single market training (the src market will set to 'xx')
  2. 'full_aug': fully uses the source market data for training
  3. 'sel_aug' : only use portion of source market data covering target market's items
  
For data sampling:
  a. 'equal'   : equally sample data from both source and target markets, providing a balanced training
  b. 'concate' : first concatenate the source and target training data, treat that a single training data
"""

import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset

sys.path.insert(1, 'src')
from model import GMF, MLP, NeuMF
from utils import *
from data import *

from tqdm import tqdm
import os
import json
import resource
import sys
import pickle


def create_arg_parser():
    parser = argparse.ArgumentParser('NeuMF_Engine')
    # Path Arguments
    parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_neg', type=int, default=4, help='number of negatives to sample during training')
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    
    # output arguments 
    parser.add_argument('--exp_name', help='name the experiment',type=str, default='exp_name')
    parser.add_argument('--exp_output', help='output results .json file',type=str, default='')
    
    # data arguments 
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify target market', type=str, default='de') # de_Electronics
    parser.add_argument('--aug_src_market', help='which data to augment with',type=str, default='xx') # us_Electronics
    
    # augmentation approaches
    # aug_method: 'no_aug', 'full_aug', 'sel_aug'
    parser.add_argument('--data_augment_method', help='how to augment data to target market',type=str, default='no_aug') 
    # sampling_method: 'concat'  'equal'
    parser.add_argument('--data_sampling_method', help='in augmentation how to sample data for training',type=str, default='concat')
    
    # MODEL selection
    parser.add_argument('--model_selection', help='which nn model to train with', type=str, default='all') # gmf, mlp, nmf
    
    # cold start setup
    parser.add_argument('--tgt_fraction', type=int, default=1, help='what fraction of data to use on target side')
    parser.add_argument('--src_fraction', type=int, default=1, help='what fraction of data to use from source side')
    
     
    return parser


"""
The main module that takes the model and dataloaders for training and testing on specific target market 
"""
def train_and_test_model(args, config, model, train_dataloader, valid_dataloader, valid_qrel, test_dataloader, test_qrel):
    opt = use_optimizer(model, config)
    loss_func = torch.nn.BCELoss()
    
    ############
    ## Train
    ############
    best_ndcg = 0.0
    best_eval_res = {}
    all_eval_res = {}
    for epoch in range(args.num_epoch):
        print('Epoch {} starts !'.format(epoch))
        model.train()
        total_loss = 0

        # train the model for some certain iterations
        train_dataloader.refresh_dataloaders()
        #iteration_num = len(train_dataloader[0])
        data_lens = [len(train_dataloader[idx]) for idx in range(train_dataloader.num_tasks)]
        iteration_num = max(data_lens)
        for iteration in range(iteration_num):
            for subtask_num in range(train_dataloader.num_tasks): # get one batch from each dataloader
                cur_train_dataloader = train_dataloader.get_iterator(subtask_num)
                try:
                    train_user_ids, train_item_ids, train_targets = next(cur_train_dataloader)
                except:
                    new_train_iterator = iter(train_dataloader[subtask_num])
                    train_user_ids, train_item_ids, train_targets = next(new_train_iterator)
                    
                if config['use_cuda'] is True:
                    train_user_ids, train_item_ids, train_targets = train_user_ids.cuda(), train_item_ids.cuda(), train_targets.cuda()
                opt.zero_grad()
                ratings_pred = model(train_user_ids, train_item_ids)
                loss = loss_func(ratings_pred.view(-1), train_targets)
                loss.backward()
                opt.step()    
                total_loss += loss.item()
        sys.stdout.flush()
        print('-' * 80)
    
    ############
    ## TEST
    ############
    #if args.model_selection=='nmf':
    valid_ov, valid_ind = test_model(model, config, valid_dataloader, valid_qrel)
    cur_ndcg = valid_ov['ndcg_cut_10']
    cur_recall = valid_ov['recall_10']
    print( f'[pytrec_based] tgt_valid: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall}')

    all_eval_res[f'valid'] = {
        'agg': valid_ov,
        'ind': valid_ind,
    }

    test_ov, test_ind = test_model(model, config, test_dataloader, test_qrel)
    cur_ndcg = test_ov['ndcg_cut_10']
    cur_recall = test_ov['recall_10']
    print( f'[pytrec_based] tgt_test: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall} \n\n')

    all_eval_res[f'test'] = {
        'agg': test_ov,
        'ind': test_ind,
    }

    return model, all_eval_res 


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)
    my_id_bank = Central_ID_Bank()
    
    ############
    ## Target Market data
    ############
    tgt_data_dir = os.path.join(args.data_dir, f'proc_data/{args.tgt_market}_5core.txt')
    print(f'loading {tgt_data_dir}')
    tgt_ratings = pd.read_csv(tgt_data_dir, sep=' ')
    
    tgt_task_generator = MAML_TaskGenerator(tgt_ratings, my_id_bank, item_thr=7, sample_df=args.tgt_fraction)
    print('loaded target data!')
    
    
    ############
    ## Source Market Data: Augmentation Approaches
    ## options: 'no_aug', 'full_aug', or 'sel_aug'
    ############
    aug_method = args.data_augment_method
    if args.aug_src_market=='us':
        src_data_dir = os.path.join(args.data_dir, f'proc_data/{args.aug_src_market}_10core.txt')
    else:
        src_data_dir = os.path.join(args.data_dir, f'proc_data/{args.aug_src_market}_5core.txt')

    if aug_method=='no_aug':
        src_task_generator = None
        args.aug_src_market = 'xx'
    if aug_method=='full_aug':
        print(f'loading {src_data_dir}')
        src_ratings = pd.read_csv(src_data_dir, sep=' ')
        src_task_generator = MAML_TaskGenerator(src_ratings, my_id_bank, item_thr=7, sample_df=args.src_fraction)
    if aug_method=='sel_aug':
        print(f'loading {src_data_dir} with limiting to target data item pool...')
        src_ratings = pd.read_csv(src_data_dir, sep=' ')
        aug_items_allowed = tgt_task_generator.item_pool_ids
        src_task_generator = MAML_TaskGenerator(src_ratings, my_id_bank, item_thr=7, items_allow=aug_items_allowed)

    sys.stdout.flush()
    
    
    ############
    ## Dataset Concatenation 
    ## options: 'equal' or 'concat' 
    ############
    print('concatenating target and source data...')
    sampling_method = args.data_sampling_method # 'concat'  'equal'

    if aug_method=='no_aug':      # 0. only use the target market train data
        task_gen_all = {
            0: tgt_task_generator,
        } 
        train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=args.num_neg, meta_split='train' )
        train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)

    elif sampling_method=='equal': # 1. equally sample from source and target 
        task_gen_all = {
            0: tgt_task_generator,
            1: src_task_generator
        } 
        train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=args.num_neg, meta_split='train' )
        train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)

    else:                         # 2. concatenate first, and then sample 
        tgt_task_dataset = tgt_task_generator.instance_a_market_train_task(0, num_negatives=args.num_neg)
        src_task_dataset = src_task_generator.instance_a_market_train_task(0, num_negatives=args.num_neg)
        train_tasksets = SingleMarket_Dataset( ConcatDataset( [tgt_task_dataset, src_task_dataset]) )
        train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)

    
    sys.stdout.flush()

    print('preparing test/valid data...')
    tgt_user_stats = tgt_task_generator.get_user_stats()
    
    tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(0, sample_batch_size=args.batch_size, shuffle=False, num_workers=0, split='valid')
    tgt_valid_qrel = tgt_task_generator.get_validation_qrel(split='valid')
    
    tgt_test_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(0, sample_batch_size=args.batch_size, shuffle=False, num_workers=0, split='test')
    tgt_test_qrel = tgt_task_generator.get_validation_qrel(split='test')
    
    
    ############
    ## Model Prepare 
    ############
    all_model_selection = ['gmf', 'mlp', 'nmf']

    results = {}

    for cur_model_selection in all_model_selection:
        sys.stdout.flush()
        args.model_selection = cur_model_selection
        config = get_model_config(args.model_selection)
        config['batch_size'] = args.batch_size
        config['optimizer'] = 'adam'
        config['use_cuda'] = args.cuda
        config['device_id'] = 0
        config['save_trained'] = True
        config['load_pretrained'] = True
        config['num_users'] = int(my_id_bank.last_user_index+1)
        config['num_items'] = int(my_id_bank.last_item_index+1)

        if args.model_selection=='gmf':
            print('model is GMF!')
            model = GMF(config)
        elif args.model_selection=='nmf':
            print('model is NeuMF!')
            model = NeuMF(config)
            if config['load_pretrained']:
                print('loading pretrained gmf and mlp...')
                model.load_pretrain_weights(args)
        else: # default is MLP
            print('model is MLP!')
            model = MLP(config)
            if config['load_pretrained']:
                print('loading pretrained gmf...')
                model.load_pretrain_weights(args)

        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            model.cuda()
        print(model)
        sys.stdout.flush()
        model, cur_model_results = train_and_test_model(args, config, model, train_dataloader, tgt_valid_dataloader, tgt_valid_qrel, tgt_test_dataloader, tgt_test_qrel)
        
        #if args.model_selection=='nmf':
        results[args.model_selection] = cur_model_results

        ############
        ## SAVE the model and idbank
        ############
        if config['save_trained']:
            model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
            save_checkpoint(model, model_dir)
            with open(cid_filename, 'wb') as centralid_file:
                pickle.dump(my_id_bank, centralid_file)
    
    
    # writing the results into a file      
    results['args'] = str(args)
    results['user_stats'] = tgt_user_stats
    with open(args.exp_output, 'w') as outfile:
        json.dump(results, outfile)
    
    print('Experiment finished success!')
    
if __name__=="__main__":
    main()
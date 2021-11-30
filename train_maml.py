import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from model import GMF, MLP, NeuMF
import learn2learn as l2l

from utils import *
from data import *
from tqdm import tqdm
import os
import json
import resource
import sys
import pickle


def create_arg_parser():
    parser = argparse.ArgumentParser('MAML_NeuMF_Engine')
    # Path Arguments
    parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size') # is kshots here
    parser.add_argument('--num_neg', type=int, default=4, help='number of negatives to sample during training')
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    
    # output arguments 
    parser.add_argument('--exp_name', help='name the experiment',type=str, default='exp_name')
    parser.add_argument('--exp_output', help='output results .json file',type=str, default='')
    
    # data arguments 
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify target market', type=str, default='de') # de_Electronics
    parser.add_argument('--aug_src_market', help='which data to augment with',type=str, default='uk') # us_Electronics
    
    # sampling_method: 'concat'  'equal'
    parser.add_argument('--data_sampling_method', help='in augmentation how to sample data for training',type=str, default='concat')
    
    #MAML arguments 
    parser.add_argument('--fast_lr', type=float, default=0.1, help='meta-learning rate') 
    # cold start setup
    parser.add_argument('--tgt_fraction', type=int, default=1, help='what fraction of data to use on target side')
    parser.add_argument('--src_fraction', type=int, default=1, help='what fraction of data to use from source side')
     
    return parser




def fast_adapt(config, batch_adapt, batch_eval, learner, loss, adaptation_steps):
    
    adapt_user_ids, adapt_item_ids, adapt_targets = batch_adapt
    eval_user_ids, eval_item_ids, eval_targets = batch_eval
    
    if config['use_cuda'] is True:
        adapt_user_ids, adapt_item_ids, adapt_targets = adapt_user_ids.cuda(), adapt_item_ids.cuda(), adapt_targets.cuda()
        eval_user_ids, eval_item_ids, eval_targets = eval_user_ids.cuda(), eval_item_ids.cuda(), eval_targets.cuda()
        
    # Adapt the model
    for step in range(adaptation_steps):
        ratings_pred = learner(adapt_user_ids, adapt_item_ids)
        train_error = loss(ratings_pred.view(-1), adapt_targets)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(eval_user_ids, eval_item_ids)
    valid_error = loss(predictions.view(-1), eval_targets)
    return valid_error



def main():
    
    parser = create_arg_parser()
#     tgt_market = 'de'
#     src_market = 'uk'
#     main_data_dir = '../DATA_FINAL_NOCAT/'
#     args = parser.parse_args(f'--data_dir {main_data_dir} --tgt_market {tgt_market} --aug_src_market {src_market} --num_epoch=15 --cuda'.split()) #
    args = parser.parse_args()
    set_seed(args)
    
    args.markets = [args.tgt_market, args.aug_src_market]
    markets_string = ''.join(args.markets)
    
    args.data_augment_method = 'full_aug'
#     args.data_sampling_method = 'concat'
    args.model_selection = 'nmf'
    
    nmf_model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
    with open(cid_filename, 'rb') as centralid_file:
        my_id_bank = pickle.load(centralid_file)

    ############
    ## All Market data
    ############
    task_gen_all = {}
    market_index = {}
    for mar_index, cur_market in enumerate(args.markets):
        cur_mkt_data_dir = os.path.join(args.data_dir, f'proc_data/{cur_market}_5core.txt')
        if cur_market=='us':
            cur_mkt_data_dir = os.path.join(args.data_dir, f'proc_data/{cur_market}_10core.txt')
        print(f'loading {cur_mkt_data_dir}')
        cur_mkt_ratings = pd.read_csv(cur_mkt_data_dir, sep=' ')
        
        cur_mkt_fraction = args.tgt_fraction
        if mar_index>=1:
            cur_mkt_fraction = args.src_fraction
            
        tgt_task_generator = MAML_TaskGenerator(cur_mkt_ratings, my_id_bank, item_thr=7, sample_df=cur_mkt_fraction)
        task_gen_all[mar_index] = tgt_task_generator
        market_index[mar_index] = cur_market

    print('loaded all data!')

    sys.stdout.flush()
    
    
    ############
    ## Dataset Concatenation 
    ############
    sampling_method = args.data_sampling_method # 'concat'  'equal'
    train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=args.num_neg, meta_split='train' )
    train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)



    print('concatenating target and source data!')
    sys.stdout.flush()

    print('preparing test/valid data...')
    
    
    task_test_all = {}
    task_valid_all = {}
    task_user_stats = {}
    for mar_index, cur_market in enumerate(args.markets):
        #user stats 
        cur_user_stats = task_gen_all[mar_index].get_user_stats()
        task_user_stats[cur_market] = cur_user_stats
        
        #valid data
        cur_valid_dataloader = task_gen_all[mar_index].instance_a_market_valid_dataloader(mar_index, sample_batch_size=args.batch_size, shuffle=True, num_workers=0, split='valid')
        cur_valid_qrel = task_gen_all[mar_index].get_validation_qrel(split='valid')
        task_valid_all[cur_market] = (iter(cur_valid_dataloader), cur_valid_dataloader, cur_valid_qrel)
        
        #test data 
        cur_test_dataloader = task_gen_all[mar_index].instance_a_market_valid_dataloader(mar_index, sample_batch_size=args.batch_size, shuffle=True, num_workers=0, split='test')
        cur_test_qrel = task_gen_all[mar_index].get_validation_qrel(split='test')
        task_test_all[cur_market] = (iter(cur_test_dataloader), cur_test_dataloader, cur_test_qrel)
        
    
    
    ############
    ## Model Prepare 
    ############
    all_model_selection = ['nmf']

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
#         config['adam_lr'] = args.lr
#         config['l2_regularization'] = args.l2_reg
        
        model = NeuMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            model.cuda()
        resume_checkpoint(model, model_dir = nmf_model_dir, device_id=config['device_id'])
        print(model)
        sys.stdout.flush()
        
        
        fast_lr = args.fast_lr  #=0.5
        meta_batch_size = train_dataloader.num_tasks #32
        adaptation_steps=1
        test_adaptation_steps = 1 #how many times adapt the model for testing time
        
        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        opt = torch.optim.Adam(maml.parameters(), lr=config['adam_lr'], weight_decay=config['l2_regularization']) 
        loss_func = torch.nn.BCELoss()
        
        
        ############
        ## Train
        ############
        for epoch in range(args.num_epoch):
            print('Epoch {} starts !'.format(epoch))
            sys.stdout.flush()
            
            # train the model for some certain iterations
            train_dataloader.refresh_dataloaders()
            data_lens = [len(train_dataloader[idx]) for idx in range(train_dataloader.num_tasks)]
            iteration_num = int(max(data_lens)/2)
            for iteration in range(iteration_num):
                opt.zero_grad()
                meta_train_loss = 0.0
                meta_valid_loss = 0.0
                for subtask_num in range(meta_batch_size): # get one batch from each dataloader
                    # Compute meta-training loss
                    #print('training maml....')
                    learner = maml.clone()   
                    cur_train_iterator = train_dataloader.get_iterator(subtask_num)
                    try:
                        adapt_batch = next(cur_train_iterator)
                        eval_batch = next(cur_train_iterator)
                    except:
                        new_train_iterator = iter(train_dataloader[subtask_num])
                        adapt_batch = next(new_train_iterator)
                        eval_batch = next(new_train_iterator)
                    
                    
                    evaluation_error = fast_adapt(config,
                                                  adapt_batch,
                                                  eval_batch,
                                                  learner,
                                                  loss_func, 
                                                  adaptation_steps)
                    
                    evaluation_error.backward()
                    meta_train_loss += evaluation_error.item()
                    
                    # Compute meta-validation loss
                    #print('evaluating maml...')
                    learner = maml.clone()
                    cur_valid_iterator = task_valid_all[ market_index[subtask_num] ][0] # get the iterator of the dataloader 
                    try:
                        adapt_batch = next(cur_valid_iterator)
                        eval_batch = next(cur_valid_iterator)
                    except:
                        new_valid_iterator = iter(task_valid_all[ market_index[subtask_num] ][1])
                        adapt_batch = next(new_valid_iterator)
                        eval_batch = next(new_valid_iterator)
                    
                    evaluation_error = fast_adapt(config,
                                                  adapt_batch,
                                                  eval_batch,
                                                  learner,
                                                  loss_func, 
                                                  adaptation_steps)
        
                    meta_valid_loss += evaluation_error.item()
    


                # Average the accumulated gradients and optimize
                for p in maml.parameters():
                    p.grad.data.mul_(1.0 / meta_batch_size)
                opt.step()    
        
        
        ############
        ## TEST
        ############
        cur_model_results = {}
        for mar_index, cur_market in enumerate(args.markets):
            # validation data 
            learner = maml.clone()
            
            for cur_test_adapt_step in range(test_adaptation_steps):
                sys.stdout.flush()
                cur_valid_iterator = task_valid_all[cur_market][0] # get the iterator of the dataloader 
                try:
                    adapt_batch = next(cur_valid_iterator)
                    eval_batch = next(cur_valid_iterator)
                except:
                    new_valid_iterator = iter(task_valid_all[cur_market][1])
                    adapt_batch = next(new_valid_iterator)
                    eval_batch = next(new_valid_iterator)
                

                evaluation_error = fast_adapt(config,
                                              adapt_batch,
                                              eval_batch,
                                              learner,
                                              loss_func, 
                                              adaptation_steps)
                
                print(f'test eval on {cur_market} adaptation step: {cur_test_adapt_step}')
                valid_ov, valid_ind = test_model(learner, config, task_valid_all[cur_market][1], task_valid_all[cur_market][2])
                cur_ndcg = valid_ov['ndcg_cut_10']
                cur_recall = valid_ov['recall_10']
                print( f'[pytrec_based] Market: {cur_market} step{cur_test_adapt_step} tgt_valid: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall}')

                cur_model_results[f'valid_{cur_market}_step{cur_test_adapt_step}'] = {
                    'agg': valid_ov,
                    'ind': valid_ind,
                }
            
                # test data  
                test_ov, test_ind = test_model(learner, config, task_test_all[cur_market][1], task_test_all[cur_market][2])
                cur_ndcg = test_ov['ndcg_cut_10']
                cur_recall = test_ov['recall_10']
                print( f'[pytrec_based] Market: {cur_market} step{cur_test_adapt_step} tgt_test: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall} \n\n')

                cur_model_results[f'test_{cur_market}_step{cur_test_adapt_step}'] = {
                    'agg': test_ov,
                    'ind': test_ind,
                }
        
        results[args.model_selection] = cur_model_results

        ############
        ## SAVE the model
        ############
        if config['save_trained']:
            #model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
            maml_nmf_output_dir = nmf_model_dir.replace('/', f'/maml{args.batch_size}_')
            save_checkpoint( maml, maml_nmf_output_dir )

    
    
    # writing the results into a file      
    results['args'] = str(args)
    results['user_stats'] = task_user_stats
    with open(args.exp_output, 'w') as outfile:
        json.dump(results, outfile)
    
    print('Experiment finished success!')
    
if __name__=="__main__":
    main()
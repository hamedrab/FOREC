import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from model import GMF, MLP, NeuMF
from model import NeuMF_MH

from utils import *
from data import *
from tqdm import tqdm
import os
import json
import resource
import sys
import pickle


def create_arg_parser():
    parser = argparse.ArgumentParser('FOREC_NeuMF_Engine')
    # Path Arguments
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_neg', type=int, default=4, help='number of negatives to sample during training')
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    
    # output arguments 
    parser.add_argument('--exp_name', help='name the experiment',type=str, default='exp_name')
    parser.add_argument('--exp_output', help='output results .json file',type=str, default='')
    
    # data arguments 
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='../DATA_FINAL_NOCAT/')
    parser.add_argument('--tgt_market', help='specify target market', type=str, default='de') 
    parser.add_argument('--aug_src_market', help='which data to augment with',type=str, default='uk') 
    

    parser.add_argument('--data_sampling_method', help='in augmentation how to sample data for training',type=str, default='concat')
    parser.add_argument('--tgt_fraction', type=int, default=1, help='what fraction of data to use on target side')
    parser.add_argument('--src_fraction', type=int, default=1, help='what fraction of data to use from source side')
     
    return parser


"""
The main module that takes the model and dataloaders for training and testing on specific target market 
"""
def train_and_test_model(args, config, model, train_dataloader, valid_dataloader, valid_qrel, test_dataloader, test_qrel, cur_tgt='xx'):
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
    #if epoch>20:
    valid_ov, valid_ind = test_model(model, config, valid_dataloader, valid_qrel)
    cur_ndcg = valid_ov['ndcg_cut_10']
    cur_recall = valid_ov['recall_10']
    print( f'[pytrec_based] {cur_tgt} tgt_valid: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall}')

    all_eval_res[f'valid_{cur_tgt}'] = {
        'agg': valid_ov,
        'ind': valid_ind,
    }

    test_ov, test_ind = test_model(model, config, test_dataloader, test_qrel)
    cur_ndcg = test_ov['ndcg_cut_10']
    cur_recall = test_ov['recall_10']
    print( f'[pytrec_based] {cur_tgt} tgt_test: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall} \n\n')

    all_eval_res[f'test_{cur_tgt}'] = {
        'agg': test_ov,
        'ind': test_ind,
    }

    return model, all_eval_res 



def freeze_model(model, allowed_fc_layers=[]):
    # freeze all the parameters 
    for param in model.parameters():
        param.requires_grad = False
    #     print(param.shape, param.requires_grad)
    
    for allowed_fc_layer in allowed_fc_layers:
        model.fc_layers[allowed_fc_layer].weight.requires_grad = True
        model.fc_layers[allowed_fc_layer].bias.requires_grad = True

    model.affine_output.weight.requires_grad = True
    model.affine_output.bias.requires_grad = True
    model.logistic.requires_grad = True
    return model





def main():
    
    parser = create_arg_parser()
#     tgt_market = 'de'
#     src_market = 'uk'
#     main_data_dir = '../DATA_FINAL_NOCAT/'
#     args = parser.parse_args(f'--data_dir {main_data_dir} --tgt_market {tgt_market} --aug_src_market {src_market} --num_epoch=15 --cuda'.split()) #
    args = parser.parse_args()
    set_seed(args)
    
    
    ############
    ## load id bank 
    ############
    args.data_augment_method = 'full_aug' #'concat'
    args.model_selection = 'nmf'
    
    nmf_model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
    with open(cid_filename, 'rb') as centralid_file:
        my_id_bank = pickle.load(centralid_file)
    
        
    # for maml model loading:
    maml_nmf_load = True
    maml_shots = 20
    nmf_model_dir = nmf_model_dir.replace('/', f'/maml{maml_shots}_')
    
    ############
    ## Train for every target market (just for demonstration purpose, in fact you only need the target market here)
    ############
    results = {}
    MH_spec = {'mh_layers':[16, 32, 16], 'adam_lr':0.01, 'l2_regularization': 0.001}
    cur_tgt_markets = [args.tgt_market, args.aug_src_market]
    for cur_tgt_market in cur_tgt_markets:
        cur_mark_fraction = args.tgt_fraction
        if cur_tgt_market==args.aug_src_market:
            cur_mark_fraction = args.src_fraction

        cur_mkt_data_dir = os.path.join(args.data_dir, f'proc_data/{cur_tgt_market}_5core.txt')
        if cur_tgt_market=='us':
            cur_mkt_data_dir = os.path.join(args.data_dir, f'proc_data/{cur_tgt_market}_10core.txt')
        print(f'loading {cur_mkt_data_dir}')
        cur_mkt_ratings = pd.read_csv(cur_mkt_data_dir, sep=' ')

        tgt_task_generator = MAML_TaskGenerator(cur_mkt_ratings, my_id_bank, item_thr=7, sample_df=cur_mark_fraction)
        task_gen_all = {
            0: tgt_task_generator,
        } 
        train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=4, meta_split='train' )
        train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)

        print('loaded target data!')
        sys.stdout.flush()


        print('preparing test/valid data...')
        tgt_user_stats = tgt_task_generator.get_user_stats()

        tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(0, sample_batch_size=args.batch_size, shuffle=False, num_workers=0, split='valid')
        tgt_valid_qrel = tgt_task_generator.get_validation_qrel(split='valid')

        tgt_test_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(0, sample_batch_size=args.batch_size, shuffle=False, num_workers=0, split='test')
        tgt_test_qrel = tgt_task_generator.get_validation_qrel(split='test')

        # load the pretrained nmf model
        nmf_config = get_model_config(args.model_selection)
        nmf_config['num_users'] = int(my_id_bank.last_user_index+1)
        nmf_config['num_items'] = int(my_id_bank.last_item_index+1)
        nmf_config['batch_size'] = args.batch_size
        nmf_config['optimizer'] = 'adam'
        nmf_config['use_cuda'] = args.cuda
        nmf_config['device_id'] = 0
        nmf_config['save_trained'] = True

        for conf_key, conf_val in MH_spec.items():
            nmf_config[conf_key] = conf_val

        model = NeuMF_MH(nmf_config)
        if nmf_config['use_cuda'] is True:
            model.cuda()

        if maml_nmf_load: 
            resume_checkpoint(model, model_dir = nmf_model_dir, device_id=nmf_config['device_id'], maml_bool=True)
        else:
            resume_checkpoint(model, model_dir = nmf_model_dir, device_id=nmf_config['device_id'])
        sys.stdout.flush()

        # freeze desired layers 
        args.unfreeze_from = -3
        if args.unfreeze_from!=0:
            cur_unfreeze_from = int(args.unfreeze_from)
            allowed_fc_layers = [idx for idx in range(cur_unfreeze_from, 0)]
            model = freeze_model(model, allowed_fc_layers=allowed_fc_layers)

        for allowed_mh_layer in range(len(model.mh_layers)):
            model.mh_layers[allowed_mh_layer].weight.requires_grad = True
            model.mh_layers[allowed_mh_layer].bias.requires_grad = True

        print('model shape and freeze status: \n')
        for param in model.parameters():
            print(param.shape, param.requires_grad)


        model, cur_model_results = train_and_test_model(args, nmf_config, model, train_dataloader, tgt_valid_dataloader, tgt_valid_qrel, tgt_test_dataloader, tgt_test_qrel, cur_tgt=cur_tgt_market )
        for k, v in cur_model_results.items():
            results[k] = v
        results[f'user_stats_{cur_tgt_market}'] = tgt_user_stats
        
        ############
        ## SAVE the model
        ############
        if config['save_trained']:
            #model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
            forec_model_output_dir = nmf_model_dir.replace('/', f'/forec{args.batch_size}_{cur_tgt_market}_')
            save_checkpoint( model, forec_model_output_dir )

    
    
    # writing the results into a file      
    results['args'] = str(args)
    with open(args.exp_output, 'w') as outfile:
        json.dump(results, outfile)
    
    print('Experiment finished success!')
    
if __name__=="__main__":
    main()
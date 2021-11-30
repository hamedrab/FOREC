"""
    Some handy functions for pytroch model training ...
"""
import torch
import sys
# sys.path.insert(1, 'qrec')
# from ConversationalMF import *
import math
import pandas as pd
import random
import pytrec_eval


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)
        return self.result

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


def get_evaluations_final(run_mf, test):
    metrics = {'recall_5', 'recall_10', 'recall_20', 'P_5', 'P_10', 'P_20', 'map_cut_10','ndcg_cut_10'}
    eval_obj = Evaluator(metrics)
    indiv_res = eval_obj.evaluate(run_mf, test)
    overall_res = eval_obj.show_all()
    return overall_res, indiv_res
    
def set_seed(args):
    random.seed(args.seed)
#     np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id, maml_bool=False):
    state_dict = torch.load(model_dir,
                        map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    
    if maml_bool:
        for key in list(state_dict.keys()):
            new_key = key.replace('module.', '')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    
    model.load_state_dict(state_dict, strict=False)


def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, network.parameters()),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer




def get_model_cid_dir(args, model_type, flip=False):
    """
    based on args and model type, this function generates idbank and checkpoint file dirs
    """
    src_market = args.aug_src_market
    tgt_market = args.tgt_market
    if flip:
        src_market = args.tgt_market 
        tgt_market = args.aug_src_market
    
    
    tmp_exp_name = f'{args.data_augment_method}_{args.data_sampling_method}'
    tmp_src_markets = src_market
    if args.data_augment_method == 'no_aug':
        #src_market = 'xx'
        tmp_exp_name = f'{args.data_augment_method}'
        tmp_src_markets = 'single'
    
    model_dir = f'checkpoints/{tgt_market}_{model_type}_{tmp_src_markets}_{tmp_exp_name}_{args.exp_name}.model'
    cid_dir = f'checkpoints/{tgt_market}_{model_type}_{tmp_src_markets}_{tmp_exp_name}_{args.exp_name}.pickle'
    return model_dir, cid_dir


def get_model_config(model_type):
    
    gmf_config = {'alias': 'gmf',
                  'adam_lr': 0.005, #1e-3,
                  'latent_dim': 8,
                  'l2_regularization': 1e-07, #0, # 0.01
                  'embedding_user': None,
                  'embedding_item': None,
                  }

    mlp_config = {'alias': 'mlp',
                  'adam_lr': 0.01, #1e-3,
                  'latent_dim': 8,
                  'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                  'l2_regularization': 1e-07, #0.0000001,  # MLP model is sensitive to hyper params
                  'pretrain': True,
                  'embedding_user': None,
                  'embedding_item': None,
                 }

    neumf_config = {'alias': 'nmf',
                    'adam_lr': 0.01, #1e-3,
                    'latent_dim_mf': 8,
                    'latent_dim_mlp': 8,
                    'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                    'l2_regularization': 1e-07, #0.0000001, #0.01,
                    'pretrain': True,
                    'embedding_user': None,
                    'embedding_item': None,
                    }
    
    config = {
      'gmf': gmf_config,
      'mlp': mlp_config,
      'nmf': neumf_config}[model_type]
    
    return config


# conduct the testing on the model
def test_model(model, config, test_dataloader, test_qrel):
    model.eval()
    task_rec_all = []
    task_unq_users = set()
    for test_batch in test_dataloader:
        test_user_ids, test_item_ids, test_targets = test_batch
        # _get_rankings function
        cur_users = [user.item() for user in test_user_ids]
        cur_items = [item.item() for item in test_item_ids]

        if config['use_cuda'] is True:
            test_user_ids, test_item_ids, test_targets = test_user_ids.cuda(), test_item_ids.cuda(), test_targets.cuda()

        with torch.no_grad():
            batch_scores = model(test_user_ids, test_item_ids)
            if config['use_cuda'] is True:
                batch_scores = batch_scores.detach().cpu().numpy()
            else:
                batch_scores = batch_scores.detach().numpy()

        for index in range(len(test_user_ids)):
            task_rec_all.append((cur_users[index], cur_items[index], batch_scores[index][0].item()))

        task_unq_users = task_unq_users.union(set(cur_users))

    task_run_mf = get_run_mf(task_rec_all, task_unq_users)
    task_ov, task_ind = get_evaluations_final(task_run_mf, test_qrel)
    #metron_ndcg, metron_recall = metron_ndcg_recall(task_run_mf, test_qrel, top_k_thr=10)
    return task_ov, task_ind


def get_run_mf(rec_list, unq_users):
    ranking = {}    
    for cuser in unq_users:
        user_ratings = [x for x in rec_list if x[0]==cuser]
        user_ratings.sort(key=lambda x:x[2], reverse=True)
        ranking[cuser] = user_ratings

    run_mf = {}
    for k, v in ranking.items():
        cur_rank = {}
        for item in v:
            cur_rank[str(item[1])]= 2+item[2]
        run_mf[str(k)] = cur_rank
    return run_mf


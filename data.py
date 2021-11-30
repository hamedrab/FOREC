import os
import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import resource


class Central_ID_Bank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """
    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        self.last_user_index = 0
        self.last_item_index = 0
        
    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            self.user_id_index[user_id] = self.last_user_index
            self.last_user_index += 1
        return self.user_id_index[user_id]
    
    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]
    
    def query_user_id(self, user_index):
        user_index_id = {v:k for k, v in self.user_id_index.items()}
        if user_index in user_index_id:
            return user_index_id[user_index]
        else:
            print(f'USER index {user_index} is not valid!')
            return 'xxxxx'
        
    def query_item_id(self, item_index):
        item_index_id = {v:k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            print(f'ITEM index {item_index} is not valid!')
            return 'yyyyy'

    
    

class MetaMarket_DataLoader(object):
    """Data Loader for a few markets, samples task and returns the dataloader for that market"""
    
    def __init__(self, task_list, sample_batch_size, task_batch_size=2, shuffle=True, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        self.num_tasks = len(task_list)
        self.task_list = task_list
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_batch_size = sample_batch_size
        self.task_list_loaders = {
            idx:DataLoader(task_list[idx], batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        self.task_batch_size = min(task_batch_size, self.num_tasks)
    
    def refresh_dataloaders(self):
        self.task_list_loaders = {
            idx:DataLoader(self.task_list[idx], batch_size=self.sample_batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=False) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        
    def get_iterator(self, index):
        return self.task_list_iters[index]
        
    def sample_task(self):
        sampled_task_idx = random.randint(0, self.num_tasks-1)
        return self.task_list_loaders[sampled_task_idx]
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, index):
        return self.task_list_loaders[index]
        
        
class MetaMarket_Dataset(object):
    """
    Wrapper around market data (task)
    ratings: {
      0: us_market_gen,
      1: de_market_gen,
      ...
    }
    """
    def __init__(self, task_gen_dict, num_negatives=4, meta_split='train'):
        self.num_tasks = len(task_gen_dict)
        if meta_split=='train':
            self.task_gen_dict = {idx:cur_task.instance_a_market_train_task(idx, num_negatives) for idx, cur_task  in task_gen_dict.items()}
        else:
            self.task_gen_dict = {idx:cur_task.instance_a_market_valid_task(idx, split=meta_split) for idx, cur_task  in task_gen_dict.items()}
        
    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        return self.task_gen_dict[index]
    

    
class SingleMarket_Dataset(object):
    """
    Wrapper around a single pytorch Dataset object
    """
    def __init__(self, mydataset):
        self.num_tasks = 1
        self.task_gen_dict = {
            0: mydataset
        }
        
    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        return self.task_gen_dict[index]
        


class MarketTask(Dataset):
    """
    Individual Market data that is going to be wrapped into a metadataset  i.e. MetaMarketDataset

    Wrapper, convert <user, item, rate> Tensor into Pytorch Dataset
    """
    def __init__(self, task_index, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.task_index = task_index
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]


    

class MAML_TaskGenerator(object):
    """Construct torch dataset"""
    
    def __init__(self, ratings, id_index_bank, item_thr=0, users_allow=None, items_allow=None, sample_df=1):
        """
        args:
            ratings: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'rate']
           
        """
        self.ratings = ratings
        self.id_index_bank = id_index_bank
        
        self.item_thr = item_thr
        self.sample_df = sample_df
        
        # filter non_allowed users and items
        if users_allow is not None:
            self.ratings = self.ratings[self.ratings['userId'].isin( users_allow )]
        if items_allow is not None:
            self.ratings = self.ratings[self.ratings['itemId'].isin( items_allow )]
        
        # get item and user pools
        self.user_pool_ids = set(self.ratings['userId'].unique())
        self.item_pool_ids = set(self.ratings['itemId'].unique())
        
        # replace ids with corrosponding index for both users and items
        self.ratings['userId'] = self.ratings['userId'].apply(lambda x: self.id_index_bank.query_user_index(x) )
        self.ratings['itemId'] = self.ratings['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x) )
        
        # get item and user pools (indexed version)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        
        # specify the splits of the data, normalize the vote
        self.user_stats = self._specify_splits()
        self.ratings['rate'] = [self.single_vote_normalize(cvote) for cvote in list(self.ratings.rate)]
        
        # create negative item samples
        self.negatives_train, self.negatives_valid, self.negatives_test = self._sample_negative( self.ratings )
        
        # split the data into train, valid, and test
        self.train_ratings, self.valid_ratings, self.test_ratings = self._split_loo( self.ratings )
        
        
    # returns how many training interation for each user has been used 
    def get_user_stats(self):
        return self.user_stats
    
    
    # adds a new column with each split, and removes the rows below the number of item_thr
    def _specify_splits(self):
        self.ratings = self.ratings.sort_values(['date'],ascending=True)
        self.ratings.reset_index(drop=True, inplace=True)
        by_userid_group = self.ratings.groupby("userId")
        
        splits = ['remove'] * len(self.ratings)
        
        user_stats = {}

        for usrid, indice in by_userid_group.groups.items():
            cur_item_list = list(indice)
            if len(cur_item_list)>= self.item_thr:
                train_up_indx = len(cur_item_list)-2
                valid_up_index = len(cur_item_list)-1
                
                sampled_train_up_indx = int(train_up_indx/self.sample_df)
        
                user_stats[usrid] = len(cur_item_list[:sampled_train_up_indx])

                for iind in cur_item_list[:sampled_train_up_indx]:
                    splits[iind] = 'train'
                for iind in cur_item_list[train_up_indx:valid_up_index]:
                    splits[iind] = 'valid'
                for iind in cur_item_list[valid_up_index:]:
                    splits[iind] = 'test'
        self.ratings['split'] = splits
        self.ratings = self.ratings[self.ratings['split']!='remove']
        self.ratings.reset_index(drop=True, inplace=True)
        
        return user_stats
    
    # ratings normalization
    def single_vote_normalize(self, cur_vote):
        if cur_vote>=1:
            return 1.0
        else:
            return 0.0
    
    
    def _split_loo(self, ratings):
        train_sp = ratings[ratings['split']=='train']
        valid_sp = ratings[ratings['split']=='valid']
        test_sp = ratings[ratings['split']=='test']
        return train_sp[['userId', 'itemId', 'rate']], valid_sp[['userId', 'itemId', 'rate']], test_sp[['userId', 'itemId', 'rate']]
    
    
    def _sample_negative(self, ratings):
        by_userid_group = self.ratings.groupby("userId")['itemId']
        negatives_train = {}
        negatives_test = {}
        negatives_valid = {}
        for userid, group_frame in by_userid_group:
            pos_itemids = set(group_frame.values.tolist())
            neg_itemids = self.item_pool - pos_itemids
            
            #neg_itemids_train = random.sample(neg_itemids, min(len(neg_itemids), 1000))
            neg_itemids_train = neg_itemids
            neg_itemids_test = random.sample(neg_itemids, min(len(neg_itemids), 99))
            neg_itemids_valid = random.sample(neg_itemids, min(len(neg_itemids), 99))
            
            negatives_train[userid] = neg_itemids_train
            negatives_test[userid] = neg_itemids_test
            negatives_valid[userid] = neg_itemids_valid
            
        return negatives_train, negatives_valid, negatives_test

                                                                    
    def instance_a_market_train_task(self, index, num_negatives, data_frac=1):
        """instance train task's torch Dataset"""
        users, items, ratings = [], [], []
        train_ratings = self.train_ratings

        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rate))
            
            cur_negs = self.negatives_train[int(row.userId)]
            cur_negs = random.sample(cur_negs, min(num_negatives, len(cur_negs)) )
            for neg in cur_negs:
                users.append(int(row.userId))
                items.append(int(neg))
                ratings.append(float(0))  # negative samples get 0 rating

        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return dataset
    
    
    def instance_a_market_train_dataloader(self, index, num_negatives, sample_batch_size, shuffle=True, num_workers=0, data_frac=1):
        """instance train task's torch Dataloader"""
        dataset = self.instance_a_market_train_task(index, num_negatives, data_frac)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        
    
    def instance_a_market_valid_task(self, index, split='valid'):
        """instance validation/test task's torch Dataset"""
        cur_ratings = self.valid_ratings
        cur_negs = self.negatives_valid
        if split.startswith('test'): 
            cur_ratings = self.test_ratings
            cur_negs = self.negatives_test
          
        users, items, ratings = [], [], []
        for row in cur_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rate))
            
            cur_uid_negs = cur_negs[int(row.userId)]
            for neg in cur_uid_negs:
                users.append(int(row.userId))
                items.append(int(neg))
                ratings.append(float(0))  # negative samples get 0 rating
            
        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return dataset
    
    def instance_a_market_valid_dataloader(self, index, sample_batch_size, shuffle=False, num_workers=0, split='valid'):
        """instance train task's torch Dataloader"""
        dataset = self.instance_a_market_valid_task(index, split=split)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


    def get_validation_qrel(self, split='valid'):
        """get pytrec eval version of qrel for evaluation"""
        cur_ratings = self.valid_ratings
        if split.startswith('test'): 
            cur_ratings = self.test_ratings
        qrel = {}
        for row in cur_ratings.itertuples():
            cur_user_qrel = qrel.get(str(row.userId), {})
            cur_user_qrel[str(row.itemId)] = int(row.rate)
            qrel[str(row.userId)] = cur_user_qrel
        return qrel   

    
    
    
 
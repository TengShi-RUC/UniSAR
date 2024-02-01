from typing import Any, List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils import const
from utils.sampler import *


class BaseDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return self.sampler.data.shape[0]

    def __getitem__(self, index) -> Any:
        return self.sampler.sample(index)

    def collate_batch(self, feed_dicts: List[dict]) -> dict:
        result_dict = dict()
        for key in feed_dicts[0].keys():
            if isinstance(feed_dicts[0][key], List):
                stack_val = list(
                    torch.tensor(list(elem))
                    for elem in zip(*[d[key] for d in feed_dicts]))
                if len(stack_val) == 1:
                    stack_val = stack_val[0]
            else:
                continue
            result_dict[key] = stack_val
        result_dict['batch_size'] = len(feed_dicts)
        result_dict['search'] = feed_dicts[0]['search']
        return result_dict


class RecDataSet(BaseDataSet):
    def __init__(self, train, user_vocab) -> None:
        super().__init__()
        if train == 'train':
            self.sampler = Sampler(data_path=const.rec_train,
                                   search=False,
                                   user_vocab=user_vocab)
        elif train == 'val':
            self.sampler = Sampler(data_path=const.rec_val,
                                   search=False,
                                   user_vocab=user_vocab)
        elif train == 'test':
            self.sampler = Sampler(data_path=const.rec_test,
                                   search=False,
                                   user_vocab=user_vocab)


class SrcDataSet(BaseDataSet):
    def __init__(self, train, user_vocab) -> None:
        super().__init__()
        if train == 'train':
            self.sampler = Sampler(data_path=const.src_train,
                                   search=True,
                                   user_vocab=user_vocab)
        elif train == 'val':
            self.sampler = Sampler(data_path=const.src_val,
                                   search=True,
                                   user_vocab=user_vocab)
        elif train == 'test':
            self.sampler = Sampler(data_path=const.src_test,
                                   search=True,
                                   user_vocab=user_vocab)


class InfoNCEDataset(Dataset):
    def __init__(self, query_vocab) -> None:
        super().__init__()
        # skip 0 for padding
        self.query_vocab = query_vocab[1:]
        if len(self.query_vocab[-1]) == 0:
            self.query_vocab = self.query_vocab[:-1]
        np.random.shuffle(self.query_vocab)
        self.item_vocab = np.array(list(range(1, const.item_id_num)))
        np.random.shuffle(self.item_vocab)

    def __len__(self) -> None:
        return 10000000000000

    def get_pad_query(self, query):
        if type(query) == str:
            query = eval(query)
        if type(query) == int:
            query = [query]
        query = query[:const.max_query_word_len]
        if len(query) < const.max_query_word_len:
            query += [0] * (const.max_query_word_len - len(query))
        return query

    def __getitem__(self, index):
        item = self.item_vocab[index % self.item_vocab.size].item()

        query = self.query_vocab[index % len(self.query_vocab)]
        query = self.get_pad_query(query)

        return {"align_neg_item": [item], "align_neg_query": [query]}

    # Collate a batch according to the list of feed dicts
    def collate_batch(self, feed_dicts: List[dict]) -> dict:
        result_dict = dict()
        for key in feed_dicts[0].keys():
            if isinstance(feed_dicts[0][key], List):
                stack_val = list(
                    torch.tensor(list(elem))
                    for elem in zip(*[d[key] for d in feed_dicts]))
                if len(stack_val) == 1:
                    stack_val = stack_val[0]
            else:
                continue
            result_dict[key] = stack_val
        return result_dict

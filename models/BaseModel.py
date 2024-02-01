import logging
import os
from typing import Dict

import torch
import torch.nn as nn

from utils import const, utils

from .Inputs import *


class BaseModel(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--dropout', type=float, default=0.1)

        return parser

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.dropout = args.dropout

        user_map_vocab = utils.load_pickle(const.user_map_vocab)
        self.user_map_vocab = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in user_map_vocab.items()
        }

        item_map_vocab = utils.load_pickle(const.item_map_vocab)
        self.item_map_vocab = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in item_map_vocab.items()
        }
        # add the mask token map
        self.item_map_vocab = self.add_mask_token(self.item_map_vocab)

        session_map_vocab = utils.load_pickle(const.session_map_vocab)
        self.session_map_vocab = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in session_map_vocab.items()
        }

        self.query_embedding = QueryFeat()
        self.session_embedding = SrcSessionFeat(
            self.query_embedding,
            ItemFeat(self.query_embedding, map_vocab=self.item_map_vocab),
            UserFeat(map_vocab=self.user_map_vocab),
            map_vocab=self.session_map_vocab)

        self.user_size = const.final_emb_size
        self.item_size = const.final_emb_size
        self.query_size = const.final_emb_size

        self.query_item_alignment = False

    def add_mask_token(self, map_vocab: Dict[str, torch.Tensor]):
        # add the mask token map
        for k, v in map_vocab.items():
            if v.dim() == 2:
                map_vocab[k] = torch.cat(
                    [v, torch.zeros((1, v.size(1)), dtype=v.dtype, device=self.device)], dim=0)
            else:
                map_vocab[k] = torch.cat(
                    [v, torch.zeros((1,), dtype=v.dtype, device=self.device)], dim=0)
        return map_vocab

    def _init_weights(self):
        # weight initialization xavier_normal (a.k.a glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                continue

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
            model_path = os.path.join(model_path, "{}.pt".format('best'))
        utils.check_dir(model_path)
        logging.info("save model to: {}".format(model_path))
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
            model_path = os.path.join(model_path, "{}.pt".format('best'))
        logging.info("load model from: {}".format(model_path))
        self.load_state_dict(torch.load(model_path, map_location=self.device))

    def count_variables(self) -> int:
        total_parameters = 0
        # logging.info(" ")  
        for name, p in self.named_parameters():
            if p.requires_grad:
                num_p = p.numel()
                total_parameters += num_p
                # logging.info("name:{} size:{} num_parameters:{}".format(
                #     name, p.size(), num_p))

        return total_parameters

    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad,
                              self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{
            'params': weight_p
        }, {
            'params': bias_p,
            'weight_decay': 0
        }]
        return optimize_dict

    def loss(self, inputs):
        if inputs['search']:
            return self.src_loss(inputs)
        else:
            return self.rec_loss(inputs)

    def predict(self, inputs):
        if inputs['search']:
            return self.src_predict(inputs)
        else:
            return self.rec_predict(inputs)

    def rec_loss(self, inputs):
        raise NotImplementedError

    def rec_predict(self, inputs):
        raise NotImplementedError

    def src_loss(self, inputs):
        raise NotImplementedError

    def src_predict(self, inputs):
        raise NotImplementedError

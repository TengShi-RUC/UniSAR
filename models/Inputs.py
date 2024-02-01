import torch
import torch.nn as nn

from utils import const


class UserFeat(nn.Module):
    def __init__(self, map_vocab=None) -> None:
        super().__init__()
        self.map_vocab = map_vocab

        self.attr_ls = const.user_feature_list
        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'{attr}_emb',
                nn.Embedding(num_embeddings=getattr(const, f'{attr}_num'),
                             embedding_dim=getattr(const, f'{attr}_dim')))
            nn.init.xavier_normal_(getattr(self, f'{attr}_emb').weight.data)
            self.size += getattr(const, f'{attr}_dim')

        self.user_trans = nn.Linear(self.size, const.final_emb_size)

    def forward(self, sample):
        feats_ls = []
        for attr in self.attr_ls:
            if attr == 'user_id':
                index = sample
            else:
                index = self.map_vocab[attr][sample]

            feats_ls.append(getattr(self, f'{attr}_emb')(index))

        return torch.tanh(self.user_trans(torch.cat(feats_ls, dim=-1))).clone()


class ItemFeat(nn.Module):
    def __init__(self, query_feat, map_vocab=None):
        super().__init__()

        self.map_vocab = map_vocab

        self.attr_ls = const.item_feature_list
        self.size = 0
        for attr in self.attr_ls:
            if attr in const.item_text_feature:
                setattr(self, f'{attr}_emb', query_feat)
                self.caption_id_emb = query_feat
                self.size += query_feat.size
            else:
                setattr(
                    self, f'{attr}_emb',
                    nn.Embedding(num_embeddings=getattr(const, f'{attr}_num'),
                                 embedding_dim=getattr(const, f'{attr}_dim'),
                                 padding_idx=0))
                nn.init.xavier_normal_(
                    getattr(self, f'{attr}_emb').weight.data)
                getattr(self, f'{attr}_emb').weight.data[0, :] = 0
                self.size += getattr(const, f'{attr}_dim')

        self.item_trans = nn.Linear(self.size, const.final_emb_size)

    def forward(self, sample):
        new_sample = sample.reshape((-1, ))
        result_emb = torch.zeros((new_sample.shape[0], const.final_emb_size),
                                 device=sample.device)
        sub_mask = new_sample != 0
        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask]

            feats_ls = []
            for attr in self.attr_ls:
                if attr == 'item_id':
                    index = sub_sample
                else:
                    index = self.map_vocab[attr][sub_sample]

                feats_ls.append(getattr(self, f'{attr}_emb')(index))
            sub_sample_emb = torch.tanh(
                self.item_trans(torch.cat(feats_ls, dim=-1))).clone()
            result_emb[sub_mask] = sub_sample_emb
        result_emb = result_emb.reshape((*sample.shape, const.final_emb_size))
        return result_emb


class QueryEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=const.word_id_num,
                                           embedding_dim=const.word_id_dim,
                                           padding_idx=0)
        nn.init.xavier_normal_(self.word_embedding.weight.data)

    def forward(self, seqs):
        seqs_mask = (seqs == 0)
        output = self.word_embedding(seqs)

        seqs_len = (~seqs_mask).sum(1, keepdim=True)
        output = output.masked_fill(seqs_mask.unsqueeze(2), 0)
        sum_emb = output.sum(dim=1)
        mean_emb = sum_emb / seqs_len

        mean_emb = mean_emb.masked_fill(seqs_len == 0, 0)

        return mean_emb.squeeze()


class QueryFeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_encoder = QueryEncoder()

        self.size = const.word_id_dim
        self.query_trans = nn.Linear(self.size, const.final_emb_size)
        self.size = const.final_emb_size

    def forward(self, sample: torch.Tensor):
        query_emb: torch.Tensor = self.query_encoder(
            sample.reshape((-1, const.max_query_word_len)))
        query_emb = query_emb.reshape((*sample.shape[:-1], -1))

        return torch.tanh(self.query_trans(query_emb)).clone()


class SrcSessionFeat(nn.Module):
    def __init__(self,
                 query_feat,
                 item_feat,
                 user_feat,
                 map_vocab=None) -> None:
        super().__init__()
        self.query_feat = query_feat
        self.item_feat = item_feat
        self.user_feat = user_feat

        self.map_vocab = map_vocab

    def get_user_emb(self, sample):
        return self.user_feat(sample)

    def get_item_emb(self, sample):
        return self.item_feat(sample)

    def get_query_emb(self, sample):
        return self.query_feat(sample)

    def forward(self, sample):
        new_sample = sample.reshape((-1, ))
        sub_mask = new_sample != 0

        result_query_emb = torch.zeros(
            (new_sample.shape[0], const.final_emb_size), device=sample.device)
        result_item_emb = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len,
             const.final_emb_size),
            device=sample.device)
        result_item_mask = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len),
            device=sample.device).bool()

        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask]
            sub_query_id = self.map_vocab['keyword'][sub_sample]
            sub_click_item_ls = self.map_vocab['pos_items'][sub_sample]
            sub_query_emb = self.get_query_emb(sub_query_id)
            sub_click_item_mask = torch.where(sub_click_item_ls == 0, 0,
                                              1).bool()
            sub_click_item_emb = self.get_item_emb(sub_click_item_ls)

            result_query_emb[sub_mask] = sub_query_emb
            result_item_emb[sub_mask] = sub_click_item_emb
            result_item_mask[sub_mask] = sub_click_item_mask

        result_query_emb = result_query_emb.reshape(
            (*sample.shape, const.final_emb_size))
        result_item_emb = result_item_emb.reshape(
            (*sample.shape, const.max_session_item_len, const.final_emb_size))
        result_item_mask = result_item_mask.reshape(
            (*sample.shape, const.max_session_item_len))

        return [result_query_emb, result_item_emb, result_item_mask]

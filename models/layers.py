import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class infoNCE(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp_init)

        self.weight_matrix = nn.Parameter(torch.randn((hdim, hdim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.tanh = nn.Tanh()

    def calculate_loss(self, query, item, neg_item):

        positive_logit = torch.sum((query @ self.weight_matrix) * item,
                                   dim=1,
                                   keepdim=True)
        negative_logits = (query @ self.weight_matrix) @ neg_item.transpose(
            -2, -1)

        positive_logit, negative_logits = self.tanh(positive_logit), self.tanh(
            negative_logits)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits),
                             dtype=torch.long,
                             device=query.device)

        return F.cross_entropy(logits / self.temp, labels, reduction='mean')

    def forward(self, query, click_item, neg_item, neg_query):
        query_loss = self.calculate_loss(query, click_item, neg_item)
        item_loss = self.calculate_loss(click_item, query, neg_query)

        return 0.5 * (query_loss + item_loss)


class feature_align(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.infoNCE_loss = infoNCE(temp_init, hdim)

    def filter_user_src_his(self, qry_his_emb, click_item_mask,
                            click_item_emb):
        qry_his_emb = qry_his_emb.unsqueeze(2).expand(-1, -1,
                                                      click_item_mask.size(2),
                                                      -1)

        src_his_query_emb = torch.masked_select(
            qry_his_emb.clone(),
            click_item_mask.unsqueeze(-1)).reshape(-1, qry_his_emb.size(-1))
        src_his_click_item_emb = torch.masked_select(click_item_emb.clone(), click_item_mask.unsqueeze(-1))\
            .reshape(-1, click_item_emb.size(-1))

        return src_his_query_emb, src_his_click_item_emb

    def forward(self, align_loss_input, query_emb, click_item_mask,
                q_click_item_emb):
        neg_item_emb, neg_query_emb = align_loss_input
        src_his_query_emb, src_his_click_item_emb = self.filter_user_src_his(
            query_emb, click_item_mask, q_click_item_emb)

        align_loss = self.infoNCE_loss(src_his_query_emb,
                                       src_his_click_item_emb, neg_item_emb,
                                       neg_query_emb)

        return align_loss


class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_unit,
                 batch_norm=False,
                 activation='relu',
                 sigmoid=False,
                 dropout=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i + 1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dims,
                 dropout,
                 output_layer=True,
                 batch_norm=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PLE_layer(nn.Module):
    def __init__(self, orig_input_dim, bottom_mlp_dims, tower_mlp_dims,
                 task_num, shared_expert_num, specific_expert_num,
                 dropout) -> None:
        super().__init__()
        self.embed_output_dim = orig_input_dim
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)

        self.task_experts = [[0] * self.task_num
                             for _ in range(self.layers_num)]
        self.task_gates = [[0] * self.task_num for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i
                                                                             -
                                                                             1]
            self.share_experts[i] = torch.nn.ModuleList([
                MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]],
                                     dropout,
                                     output_layer=False,
                                     batch_norm=False)
                for k in range(self.shared_expert_num)
            ])
            self.share_gates[i] = torch.nn.Sequential(
                torch.nn.Linear(
                    input_dim,
                    shared_expert_num + task_num * specific_expert_num),
                torch.nn.Softmax(dim=1))
            for j in range(task_num):
                self.task_experts[i][j] = torch.nn.ModuleList([
                    MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]],
                                         dropout,
                                         output_layer=False,
                                         batch_norm=False)
                    for k in range(self.specific_expert_num)
                ])
                self.task_gates[i][j] = torch.nn.Sequential(
                    torch.nn.Linear(input_dim,
                                    shared_expert_num + specific_expert_num),
                    torch.nn.Softmax(dim=1))
            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)

        self.tower = torch.nn.ModuleList([
            MultiLayerPerceptron(bottom_mlp_dims[-1],
                                 tower_mlp_dims,
                                 dropout,
                                 output_layer=False,
                                 batch_norm=False) for i in range(task_num)
        ])

    def forward(self, emb):
        task_fea = [emb for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            share_output = [
                expert(task_fea[-1]).unsqueeze(1)
                for expert in self.share_experts[i]
            ]
            task_output_list = []
            for j in range(self.task_num):
                task_output = [
                    expert(task_fea[j]).unsqueeze(1)
                    for expert in self.task_experts[i][j]
                ]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num - 1:  # 最后一层不需要计算share expert 的输出
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)

        results = [
            self.tower[i](task_fea[i]).squeeze(1) for i in range(self.task_num)
        ]
        return results

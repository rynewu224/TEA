import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.conv import GATConv
import numpy as np

class FFN(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FFN, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TEA_metapath_v1(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(TEA_metapath_v1, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.edim = edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args
        self.aggr_type = args.aggr_type

        # Self-Attention Block, encode user behavior sequences
        num_heads = 1
        self.item_attn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_attn_layer = nn.MultiheadAttention(edim, num_heads, args.droprate)
        self.item_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_ffn = FFN(edim, args.droprate)
        self.item_last_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.seq_lin = nn.Linear(edim + edim + edim, edim)

        # RNN Block, encode user neighbors hist item
        self.neighbor_rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)
        self.metapath_rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)

        # GNN Block, encode social information
        if self.aggr_type == 'sage':
            pass
        elif self.aggr_type == 'gat':
            self.user_attn0 = nn.Linear(edim + edim, edim, bias=False)
            self.user_attn1 = nn.Linear(edim, 1, bias=False)
            self.item_attn0 = nn.Linear(edim + edim, edim, bias=False)
            self.item_attn1 = nn.Linear(edim, 1, bias=False)
        else:
            raise NotImplemented(f'Invalid gnn type: {self.aggr_type}')

        # Fuse Layer
        self.nbr_item_fsue_lin = nn.Linear(edim + edim, edim)
        self.nbr_ffn_layernom = nn.LayerNorm(edim, eps=1e-8)
        self.nbr_ffn = FFN(edim, args.droprate)
        self.nbr_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.posn_embs = nn.Embedding(args.seq_maxlen, edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight[1:], a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight[1:], a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.posn_embs.weight[1:], a=-0.5 / args.seq_maxlen, b=0.5 / args.seq_maxlen)

        self.act = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(args.droprate)

    def seq2feat(self, seq_iid):
        batch_size, seq_maxlen = seq_iid.shape
        timeline_mask = torch.BoolTensor(seq_iid == 0).to(self.dev)  # mask the padding item
        seqs = self.item_embs(seq_iid.to(self.dev)) * (self.item_embs.embedding_dim ** 0.5)  # Rescale emb
        positions = np.tile(np.array(range(seq_maxlen), dtype=np.int64), [batch_size, 1])
        seqs += self.posn_embs(torch.LongTensor(positions).to(self.dev))
        seqs = self.dropout(seqs)

        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        seqs = torch.transpose(seqs, 0, 1)  # seqlen x B x d
        query = self.item_attn_layernorm(seqs)
        mha_outputs, _ = self.item_attn_layer(query, seqs, seqs, attn_mask=attention_mask)

        seqs = query + mha_outputs
        seqs = torch.transpose(seqs, 0, 1)  # B x seqlen x d
        seqs = self.item_ffn_layernorm(seqs)
        seqs = self.item_ffn(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)  # B x seqlen x d
        seqs = self.item_last_layernorm(seqs)

        return seqs

    def nbr2feat(self, uid, nbr, nbr_iid):
        if self.aggr_type == 'sage':
            # Get mask and neighbors length
            batch_size, nbr_maxlen, seq_maxlen = nbr_iid.shape
            nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
            nbr_seq_mask = torch.BoolTensor(nbr_iid == 0).to(self.dev)  # B x nl x sl
            nbr_len = (nbr_maxlen - nbr_mask.sum(1))  # B
            nbr_len[torch.where(nbr_len == 0)] = 1.0  # to avoid deivide by zero

            # Get embs
            # uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
            nbr = nbr.to(self.dev)  # B x nl
            nbr_iid = nbr_iid.to(self.dev)  # B x nl x sl
            # user_emb = self.dropout(self.user_embs(uid))  # B x  1 x d
            nbr_emb = self.dropout(self.user_embs(nbr))  # B x nl x d
            nbr_item_emb = self.dropout(self.item_embs(nbr_iid))  # B x nl x sl x d

            # Static Social Network Features
            nbr_emb *= ~nbr_mask.unsqueeze(-1)  # B x nl x d
            nbr_len = nbr_len.view(batch_size, 1, 1)  # B x 1 x 1
            nbr_feat = nbr_emb.sum(dim=1, keepdim=True) / nbr_len  # B x 1 x d

            # Temporal Neighbor-Items Features
            nbr_seq_mask = nbr_seq_mask.unsqueeze(-1)  # B x nl x sl x 1
            nbr_seq_mask = nbr_seq_mask.permute(0, 2, 1, 3)  # B x sl x nl x 1
            nbr_item_emb = nbr_item_emb.permute(0, 2, 1, 3)  # B x sl x nl x d
            nbr_item_emb *= ~nbr_seq_mask  # B x sl x nl x d
            nbr_seq_len = (seq_maxlen - nbr_seq_mask.sum(dim=2))  # B x sl x 1
            nbr_seq_len[torch.where(nbr_seq_len == 0)] = 1.0  # to avoid deivide by zero
            nbr_seq_feat = nbr_item_emb.sum(dim=2) / nbr_seq_len  # B x sl x d
            nbr_seq_feat, _ = self.neighbor_rnn(nbr_seq_feat)  # B x sl x d

        elif self.aggr_type == 'gat':
            # Get masks
            nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
            nbr_seq_mask = torch.BoolTensor(nbr_iid == 0).to(self.dev)  # B x nl x sl

            # Get embs
            uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
            nbr = nbr.to(self.dev)  # B x nl
            nbr_iid = nbr_iid.to(self.dev)  # B x nl x sl
            user_emb = self.dropout(self.user_embs(uid))  # B x  1 x d
            nbr_emb = self.dropout(self.user_embs(nbr))  # B x nl x d
            nbr_item_emb = self.dropout(self.item_embs(nbr_iid))  # B x nl x sl x d

            # Static Social Network Features
            user_attn_self_emb = user_emb.expand_as(nbr_emb)  # B x nl x d
            user_attn_h = self.user_attn1(self.leaky_relu(self.dropout(self.item_attn0(
                torch.cat([user_attn_self_emb, nbr_emb], dim=-1)))))  # B x nl x 1
            user_attn_score = user_attn_h + -1e9 * nbr_mask.unsqueeze(-1)
            user_attn_a = F.softmax(user_attn_score, dim=1)
            nbr_feat = (user_attn_a * nbr_emb).sum(dim=1, keepdims=True)  # B x 1 x d

            # Temporal Neighbor-Items Features
            item_attn_self_emb = user_emb.unsqueeze(1).expand_as(nbr_item_emb)  # B x nl x sl x d
            item_attn_h = self.item_attn1(self.leaky_relu(self.dropout(self.item_attn0(
                torch.cat([item_attn_self_emb, nbr_item_emb], dim=-1)))))  # B x nl x sl x 1
            item_attn_score = item_attn_h + -1e9 * nbr_seq_mask.unsqueeze(-1)
            item_attn_a = F.softmax(item_attn_score, dim=1)
            nbr_seq_feat = (item_attn_a * nbr_item_emb).sum(dim=1)  # B x sl x d

        # GRU
        nbr_seq_feat, _ = self.neighbor_rnn(nbr_seq_feat)  # B x sl x d

        nbr_feat = nbr_feat.expand_as(nbr_seq_feat)
        nbr_feat = self.nbr_item_fsue_lin(torch.cat([nbr_feat, nbr_seq_feat], dim=-1))
        nbr_feat = self.nbr_ffn_layernom(nbr_feat)
        nbr_feat = self.nbr_ffn(nbr_feat)
        nbr_feat = self.nbr_last_layernorm(nbr_feat)

        return nbr_feat

    def metapath2feat(self, uid, seq, metapath):

        # print(f'uid={uid.shape}', uid[:5])
        # print(f'seq={seq.shape}', seq[:5])
        # print(f'meta={meta.shape}', meta[:5])
        batch_size, seq_len, meta_maxlen = metapath.shape

        # print("st uid", uid.shape,'seq', seq.shape,'meta', metapath.shape)

        uid = uid.unsqueeze(-1).unsqueeze(-1)  # B * 1 * 1
        seq = seq.unsqueeze(-1)  # B * sl *1

        meta_mask = torch.BoolTensor(metapath != 0).sum(-1).to(self.dev)  # B*sl
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))  # B*1*1*dim
        seq_emb = self.dropout(self.item_embs(seq.to(self.dev)))  # B*sl*1*dim
        meta_emb = self.dropout(self.user_embs(metapath.to(self.dev))).unsqueeze(-2)  # B * sl * meta_maxlen * 1 * dim

        # print('emb user',user_emb.shape,'seq',seq_emb.shape,'meta',meta_emb.shape)

        meta_path_emb = torch.cat(
            [seq_emb, user_emb.expand_as(seq_emb)], dim=-2) \
            .unsqueeze(-3).expand(-1, -1, meta_maxlen, -1, -1)  # B*sl*1*2*dim
        # print('first_meta_path_emb', meta_path_emb.shape)

        meta_path_emb = torch.cat((meta_emb, meta_path_emb), dim=-2)  # B*sl*meta_maxlen*3*dim
        # metapath(target_user - seq_item - nbr_user) length = 3
        # print('second_meta_path_emb', meta_path_emb.shape)

        meta_path_emb = meta_path_emb.view(-1, 3, self.edim)
        meta_path_emb, _ = self.metapath_rnn(meta_path_emb)
        meta_path_emb = meta_path_emb.view(batch_size, seq_len, meta_maxlen, 3, -1)

        tmp_user_emb = meta_path_emb[:, :, :, -1, :].sum(1).sum(1)  # B*dim
        # print('tmp_user_emb',tmp_user_emb.shape)
        user_mask = meta_mask.sum(-1, keepdim=True)  # B*1
        # print('user mask', user_mask.shape)
        user_mask[torch.where(user_mask == 0)] = 1.0
        fin_user_emb = tmp_user_emb / user_mask  # B*dim
        # print('fin_user_emb',fin_user_emb.shape)

        return fin_user_emb

    def dual_pred(self, seq_hu, nbr_hu, hi):
        seq_logits = (seq_hu * hi).sum(dim=-1)
        nbr_logits = (nbr_hu * hi).sum(dim=-1)
        return seq_logits + nbr_logits

    def dual_forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid, metapath = batch

        # 1. Embedding layer
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))
        # seq_emb = self.dropout(self.item_embs(seq.to(self.dev)))
        pos_item_emb = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_item_emb = self.dropout(self.item_embs(neg.to(self.dev)))
        # nbr_user_emb = self.dropout(self.user_embs(nbr.to(self.dev)))
        # nbr_item_emb = self.dropout(self.item_embs(nbr_iid.to(self.dev)))
        # meta_emb = self.dropout(self.user_embs(metapath.to(self.dev)))

        # 2. Encode user behavior sequence
        seq_feat = self.seq2feat(seq)  # B x sl x d

        # 3. Encode metapath [target_user - seq_item - nbr_user]
        metapath_user_emb = self.metapath2feat(uid, seq, metapath)
        metapath_user_emb = metapath_user_emb.unsqueeze(1).expand_as(seq_feat)
        user_emb = user_emb.unsqueeze(1).expand_as(seq_feat)
        seq_feat = self.seq_lin(torch.cat([seq_feat, metapath_user_emb, user_emb], dim=-1))

        # 4. Propagate user intent to his neighbors through time
        nbr_feat = self.nbr2feat(uid, nbr, nbr_iid)  # B x sl x d

        # 5. Calculate user-item scores
        pos_logits = self.dual_pred(seq_feat, nbr_feat, pos_item_emb)  # B x sl
        pos_logits = pos_logits.unsqueeze(-1)  # B x sl x 1

        seq_feat = seq_feat.unsqueeze(-2).expand_as(neg_item_emb)  # B x sl x ns x d
        nbr_feat = nbr_feat.unsqueeze(-2).expand_as(neg_item_emb)  # B x sl x ns x d
        neg_logits = self.dual_pred(seq_feat, nbr_feat, neg_item_emb)  # B x sl x ns

        return pos_logits, neg_logits, user_emb, pos_item_emb, neg_item_emb

    def get_parameters(self):
        param_list = [
            {'params': self.item_attn_layernorm.parameters()},
            {'params': self.item_attn_layer.parameters()},
            {'params': self.item_ffn_layernorm.parameters()},
            {'params': self.item_ffn.parameters()},
            {'params': self.item_last_layernorm.parameters()},
            {'params': self.seq_lin.parameters()},

            {'params': self.nbr_item_fsue_lin.parameters()},
            {'params': self.nbr_ffn_layernom.parameters()},
            {'params': self.nbr_ffn.parameters()},
            {'params': self.nbr_last_layernorm.parameters()},

            {'params': self.neighbor_rnn.parameters()},
            {'params': self.metapath_rnn.parameters()},

            {'params': self.user_embs.parameters(), 'weight_decay': 0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0},
            {'params': self.posn_embs.parameters(), 'weight_decay': 0},
        ]

        if self.aggr_type == 'gat':
            param_list.extend([
                {'params': self.user_attn0.parameters()},
                {'params': self.user_attn1.parameters()},
                {'params': self.item_attn0.parameters()},
                {'params': self.item_attn1.parameters()},
            ])

        return param_list

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid, meta = eval_batch
                uid = uid.long()
                seq = seq.long()
                nbr = nbr.long()
                nbr_iid = nbr_iid.long()
                eval_iid = eval_iid.long()
                item_emb = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                seq_feat = self.seq2feat(seq)[:, -1, :]

                meta_user_emb = self.metapath2feat(uid, seq, meta)
                user_emb = self.user_embs(uid.to(self.dev))
                seq_feat = self.seq_lin(torch.cat([seq_feat, meta_user_emb, user_emb], dim=-1))

                nbr_feat = self.nbr2feat(uid, nbr, nbr_iid)[:, -1, :]
                seq_feat = seq_feat.unsqueeze(1).expand_as(item_emb)
                nbr_feat = nbr_feat.unsqueeze(1).expand_as(item_emb)
                batch_score = self.dual_pred(seq_feat, nbr_feat, item_emb)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

# ---------------------------------------


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h[out_key])


class HGTRec(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGTRec, self).__init__()
        self.G = G
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.embedding_layer = nn.ParameterDict()
        self.adapt_ws = nn.ModuleDict()

        for ntype in G.ntypes:
            emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), n_hid), requires_grad=True)
            nn.init.uniform_(emb, a=-0.5 / G.number_of_nodes(ntype), b=0.5 / G.number_of_nodes(ntype))
            self.embedding_layer[ntype] = emb
            self.adapt_ws[ntype] = nn.Linear(n_inp, n_hid)

        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))


    def get_emebeddings(self):
        h = {}

        # Build input
        for ntype in self.G.ntypes:
            self.G.nodes[ntype].data['inp'] = self.embedding_layer[ntype]
            h[ntype] = F.gelu(self.adapt_ws[ntype](self.G.nodes[ntype].data['inp']))

        # Graph conv
        for i in range(self.n_layers):
            h = self.gcs[i](self.G, h)

        return h

    def pred(self, hu, hi):
        logits = (hu * hi).sum(dim=-1)
        return logits

    def forward(self, batch):
        uid, pos, neg = batch

        h = self.get_emebeddings()
        user_emb = h['user'][uid]  # B x D
        pos_item_emb = h['item'][pos]  # B x L x D
        neg_item_emb = h['item'][neg]  # B x L x Neg x D

        user_emb = user_emb.unsqueeze(1).expand_as(pos_item_emb)  # B x L x D
        pos_logits = self.pred(user_emb, pos_item_emb)  # B x L
        pos_logits = pos_logits.unsqueeze(-1)  # B x L x 1

        user_emb = user_emb.unsqueeze(2).expand_as(neg_item_emb)  # B x L x Neg x D
        neg_logits = self.pred(user_emb, neg_item_emb)  # B x L x Neg

        return pos_logits, neg_logits, user_emb, pos_item_emb, neg_item_emb

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            h = self.get_emebeddings()
            dev = h['user'].device
            for i, eval_batch in enumerate(eval_loader):
                uid, _, _, _, eval_iid, _ = eval_batch
                uid = uid.long().to(dev)
                eval_iid = eval_iid.long().to(dev)

                item_emb = F.embedding(eval_iid, h['item'], padding_idx=0)  # B x L x D
                user_emb = F.embedding(uid, h['user'], padding_idx=0)  # B x D
                user_emb = user_emb.unsqueeze(1).expand_as(item_emb)  # B x L x D
                batch_score = self.pred(user_emb, item_emb)  # B x L
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

# ---------------------------------------

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # meta_paths = [['pa', 'ap'], ['pf', 'fp']]

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, meta_edge_droprates):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            # meta_paths = [['pa', 'ap'], ['pf', 'fp']]
            for meta_path, meta_edge_droprate in zip(self.meta_paths, meta_edge_droprates):
                new_g = dgl.metapath_reachable_graph(g, meta_path)
                print(meta_path, 'num_edges:', new_g.num_edges(), end=' ')
                if meta_edge_droprate > 0.0:
                    num_egdes = new_g.num_edges()
                    remove_eid = np.random.choice(range(num_egdes), int(num_egdes * meta_edge_droprate), replace=False)
                    new_g = dgl.remove_edges(new_g, remove_eid)
                print('keep_edges:', new_g.num_edges())
                new_g = new_g.to('cuda:0')
                self._cached_coalesced_graph[meta_path] = new_g



        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        # meta_paths = [['pa', 'ap'], ['pf', 'fp']]

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

class HANRec(nn.Module):
    def __init__(self, G, user_meta_paths, item_meta_paths,
                 in_size, hidden_size, out_size,
                 num_heads, dropout,
                 user_meta_edge_droprates,
                 item_meta_edge_droprates,
                 ):
        super(HANRec, self).__init__()
        self.G = G
        self.user_meta_edge_droprates = user_meta_edge_droprates
        self.item_meta_edge_droprates = item_meta_edge_droprates
        self.user_layers = nn.ModuleList()
        self.user_layers.append(HANLayer(user_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.user_layers.append(HANLayer(user_meta_paths, hidden_size * num_heads[l - 1],
                                             hidden_size, num_heads[l], dropout))

        self.item_layers = nn.ModuleList()
        self.item_layers.append(HANLayer(item_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.item_layers.append(HANLayer(item_meta_paths, hidden_size * num_heads[l - 1],
                                             hidden_size, num_heads[l], dropout))

        self.user_embs = nn.Parameter(torch.Tensor(G.number_of_nodes('user'), hidden_size), requires_grad=True)
        nn.init.uniform_(self.user_embs, a=-0.5 / G.number_of_nodes('user'), b=0.5 / G.number_of_nodes('user'))

        self.item_embs = nn.Parameter(torch.Tensor(G.number_of_nodes('item'), hidden_size), requires_grad=True)
        nn.init.uniform_(self.user_embs, a=-0.5 / G.number_of_nodes('item'), b=0.5 / G.number_of_nodes('item'))

    def get_emebeddings(self):

        h_user = self.user_embs
        for gnn in self.user_layers:
            h_user = gnn(self.G, h_user, self.user_meta_edge_droprates)

        h_item = self.item_embs
        for gnn in self.item_layers:
            h_item = gnn(self.G, h_item, self.item_meta_edge_droprates)

        return {'user': h_user, 'item': h_item}

    def pred(self, hu, hi):
        logits = (hu * hi).sum(dim=-1)
        return logits

    def forward(self, batch):
        uid, pos, neg = batch

        h = self.get_emebeddings()
        user_emb = h['user'][uid]  # B x D
        pos_item_emb = h['item'][pos]  # B x L x D
        neg_item_emb = h['item'][neg]  # B x L x Neg x D

        user_emb = user_emb.unsqueeze(1).expand_as(pos_item_emb)  # B x L x D
        pos_logits = self.pred(user_emb, pos_item_emb)  # B x L
        pos_logits = pos_logits.unsqueeze(-1)  # B x L x 1

        user_emb = user_emb.unsqueeze(2).expand_as(neg_item_emb)  # B x L x Neg x D
        neg_logits = self.pred(user_emb, neg_item_emb)  # B x L x Neg

        return pos_logits, neg_logits, user_emb, pos_item_emb, neg_item_emb

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            h = self.get_emebeddings()
            dev = h['user'].device
            for i, eval_batch in enumerate(eval_loader):
                uid, _, _, _, eval_iid, _ = eval_batch
                uid = uid.long().to(dev)
                eval_iid = eval_iid.long().to(dev)

                item_emb = F.embedding(eval_iid, h['item'], padding_idx=0)  # B x L x D
                user_emb = F.embedding(uid, h['user'], padding_idx=0)  # B x D
                user_emb = user_emb.unsqueeze(1).expand_as(item_emb)  # B x L x D
                batch_score = self.pred(user_emb, item_emb)  # B x L
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

# ---------------------------------------


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G, out_key):
        input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        h_dict = self.layer1(G, input_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get appropriate logits
        return h_dict[out_key]


class RGCNRec(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, droprate=0.5):
        super(RGCNRec, self).__init__()
        self.G = G
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.LeakyReLU()
        self.embedding_layer = nn.ParameterDict()
        for ntype in G.ntypes:
            emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size), requires_grad=True)
            nn.init.uniform_(emb, a=-0.5 / G.number_of_nodes(ntype), b=0.5 / G.number_of_nodes(ntype))
            self.embedding_layer[ntype] = emb

    def get_emebeddings(self):
        for ntype in self.G.ntypes:
            self.G.nodes[ntype].data['inp'] = self.embedding_layer[ntype]

        h = {ntype: self.G.nodes[ntype].data['inp'] for ntype in self.G.ntypes}
        h = {k: self.dropout(v) for k, v in h.items()}
        h = self.layer1(self.G, h)
        h = {k: self.dropout(self.act(v)) for k, v in h.items()}
        h = self.layer2(self.G, h)
        return h

    def pred(self, hu, hi):
        logits = (hu * hi).sum(dim=-1)
        return logits

    def forward(self, batch):
        uid, pos, neg = batch

        h = self.get_emebeddings()
        user_emb = h['user'][uid]  # B x D
        pos_item_emb = h['item'][pos]  # B x L x D
        neg_item_emb = h['item'][neg]  # B x L x Neg x D

        user_emb = user_emb.unsqueeze(1).expand_as(pos_item_emb)  # B x L x D
        pos_logits = self.pred(user_emb, pos_item_emb)  # B x L
        pos_logits = pos_logits.unsqueeze(-1)  # B x L x 1

        user_emb = user_emb.unsqueeze(2).expand_as(neg_item_emb)  # B x L x Neg x D
        neg_logits = self.pred(user_emb, neg_item_emb)  # B x L x Neg

        return pos_logits, neg_logits, user_emb, pos_item_emb, neg_item_emb

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            h = self.get_emebeddings()
            dev = h['user'].device
            for i, eval_batch in enumerate(eval_loader):
                uid, _, _, _, eval_iid, _ = eval_batch
                uid = uid.long().to(dev)
                eval_iid = eval_iid.long().to(dev)

                item_emb = F.embedding(eval_iid, h['item'], padding_idx=0)  # B x L x D
                user_emb = F.embedding(uid, h['user'], padding_idx=0)  # B x D
                user_emb = user_emb.unsqueeze(1).expand_as(item_emb)  # B x L x D
                batch_score = self.pred(user_emb, item_emb)  # B x L
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

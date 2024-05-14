
import torch.nn


from HyperGraph.layers.HGTransformer import encoder



def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if not torch.sum(cond):
            break
        t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t


class HypaerGraph_Model(torch.nn.Module):
    def __init__(self, config):
        super(HypaerGraph_Model, self).__init__()
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']

        self._emb_size = config['hidden_size']
        self._intermediate_size = config['intermediate_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']

        self._voc_size = config['vocab_size']
        # self._hg_conv_layer = config['hg_conv_layer']
        self._edge_num = config['max_hyperedge_num']
        self._node_num = config['max_seq_len']
        self._n_relation = config['num_relations']
        self._n_edge = config['num_edges']
        self._use_disentangle = config['use_disentangle']
        self._use_encoder = config['use_decoder']
        self._use_dynamic = config['use_dynamic']
        self._use_global = config['use_global']
        self._use_local = config['use_local']
        self._use_node_heterogeneity = config['use_node_heterogeneity']

        self._e_soft_label = config['entity_soft_label']
        self._r_soft_label = config['relation_soft_label']

        self._lambda = config['lambda']
        self._beta = config['beta']
        self._gamma = config['gamma']

        self._device = config["device"]

        # INIT EMBEDDING
        node_embeddings = torch.nn.Embedding(self._voc_size, self._emb_size)
        node_embeddings.weight.data = truncated_normal(node_embeddings.weight.data, std=0.02)
        self.node_embedding = node_embeddings
        self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=self._emb_size, eps=1e-12, elementwise_affine=True)

        if self._use_node_heterogeneity:
            self.node_embedding_k = torch.nn.Embedding(self._n_edge, self._emb_size)
            self.node_embedding_k.weight.data = truncated_normal(self.node_embedding_k.weight.data, std=0.02)
            # torch.nn.init.xavier_uniform_(self.node_embedding_k.weight)
            self.node_embedding_v = torch.nn.Embedding(self._n_edge, self._emb_size)
            self.node_embedding_v.weight.data = truncated_normal(self.node_embedding_v.weight.data, std=0.02)
            # torch.nn.init.xavier_uniform_(self.node_embedding_v.weight)

        if self._use_local:
            # Hypergraph encoder layers
            self.encoder_model = encoder(
            # self.encoder_model = transformerEncoder(
                n_layer=self._n_layer,
                n_head=self._n_head,
                d_key=self._emb_size // self._n_head,
                d_value=self._emb_size // self._n_head,
                d_model=self._emb_size,
                d_inner_hid=self._intermediate_size,
                prepostprocess_dropout=self._prepostprocess_dropout,
                attention_dropout=self._attention_dropout)

        if self._use_global:
            self.global_encoder_model = encoder(
                # self.encoder_model = transformerEncoder(
                n_layer=self._n_layer,
                n_head=self._n_head,
                d_key=self._emb_size // self._n_head,
                d_value=self._emb_size // self._n_head,
                d_model=self._emb_size,
                d_inner_hid=self._intermediate_size,
                prepostprocess_dropout=self._prepostprocess_dropout,
                attention_dropout=self._attention_dropout)
            # Double view linear layer
            self.concat_D = torch.nn.Linear(self._emb_size * 2, self._emb_size)
            self.concat_D.weight.data = truncated_normal(self.concat_D.weight.data, std=0.02)



        self.fc1 = torch.nn.Linear(self._emb_size, self._emb_size)
        self.fc1.weight.data = truncated_normal(self.fc1.weight.data, std=0.02)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=self._emb_size, eps=1e-7, elementwise_affine=True)
        self.fc2_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._voc_size)), 0.0)

        self.link_prediction_loss = SoftmaxWithCrossEntropy()

    def forward(self, data, is_train=False):
        input_ids, input_mask, mask_pos, mask_label, mask_type, \
            incidence_matrix_T, node_num, hyperedge_num, mask_edge_index, node_type, \
            g_input_ids, g_incidence_matrix, g_node_type = \
                data[0], data[1], data[2], data[3], data[4], \
                data[5], data[6], data[7], data[8], data[9], data[10], \
                data[11], data[12]
        self._device = input_ids.device
        batch_size = input_mask.size(0)

        if self._use_local:
            # Embedding
            emb_out, nodes_key, nodes_value, n_head_self_attn_mask = self.embedding(input_ids, node_type, input_mask)
            # Hyperedge initialize
            e = self.hyperedge_init(incidence_matrix_T, batch_size)
            # Encoder
            _enc_out, e = self.encoder_model(
                enc_input=emb_out,
                nodes_key=nodes_key,
                nodes_value=nodes_value,
                adj_matrix=incidence_matrix_T.transpose(1, 2),
                attn_bias=n_head_self_attn_mask,
                e=e
            )
            # Hypergraph restructure
            if self._use_encoder:
                local_re_err = self.decoder(_enc_out, e, incidence_matrix_T)

        if self._use_global:
            # embedding
            g_emb_out, g_nodes_key, g_nodes_value, g_n_head_self_attn_mask = self.embedding(g_input_ids, g_node_type, input_mask)
            # hyperedge initialize
            g_e = self.hyperedge_init(g_incidence_matrix, batch_size)
            # Get node embeddings
            g_enc_out, g_e = self.global_encoder_model(
                enc_input=g_emb_out,
                nodes_key=g_nodes_key,
                nodes_value=g_nodes_value,
                adj_matrix=g_incidence_matrix.transpose(1, 2),
                attn_bias=g_n_head_self_attn_mask,
                e=g_e
            )
            # Hypergraph restructure
            if self._use_encoder:
                global_re_err = self.decoder(g_enc_out, g_e, g_incidence_matrix)


        mask_pos = mask_pos.unsqueeze(1)
        mask_pos = mask_pos[:, :, None].expand(-1, -1, self._emb_size)
        if self._use_local:
            h_masked = torch.gather(input=_enc_out, dim=1, index=mask_pos).reshape([-1, _enc_out.size(-1)])
        elif self._use_global:
            h_masked = torch.gather(input=g_enc_out, dim=1, index=mask_pos).reshape([-1, g_enc_out.size(-1)])

        if self._use_global and self._use_local:
            g_h_masked = torch.gather(input=g_enc_out, dim=1, index=mask_pos).reshape([-1, g_enc_out.size(-1)])
            h_masked = self.concat_D(torch.cat((h_masked, g_h_masked), dim=-1))


        h_masked = self.fc1(h_masked)
        h_masked = torch.nn.GELU()(h_masked)

        h_masked = self.layer_norm2(h_masked)

        fc_out = torch.nn.functional.linear(h_masked, self.node_embedding.weight, self.fc2_bias)

        special_indicator = torch.empty(input_ids.size(0), 2).to(self._device)
        torch.nn.init.constant_(special_indicator, -1)
        relation_indicator = torch.empty(input_ids.size(0), self._n_relation).to(self._device)
        torch.nn.init.constant_(relation_indicator, -1)
        entity_indicator = torch.empty(input_ids.size(0), (self._voc_size - self._n_relation - 2)).to(self._device)
        torch.nn.init.constant_(entity_indicator, 1)
        type_indicator = torch.cat((relation_indicator, entity_indicator), dim=1).to(self._device)
        mask_type = mask_type.unsqueeze(1)
        type_indicator = torch.mul(type_indicator, mask_type)
        type_indicator = torch.cat([special_indicator, type_indicator], dim=1)
        type_indicator = torch.nn.functional.relu(type_indicator)

        fc_out_mask = 1000000.0 * (type_indicator - 1.0)
        fc_out = torch.add(fc_out, fc_out_mask)

        one_hot_labels = torch.nn.functional.one_hot(mask_label, self._voc_size)
        type_indicator = torch.sub(type_indicator, one_hot_labels)
        num_candidates = torch.sum(type_indicator, dim=1)

        soft_labels = ((1 + mask_type) * self._e_soft_label +
                       (1 - mask_type) * self._r_soft_label) / 2.0
        soft_labels = soft_labels.expand(-1, self._voc_size)
        soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * \
                      torch.mul(type_indicator, 1.0 / torch.unsqueeze(num_candidates, 1))

        link_prediction_loss = self.link_prediction_loss(logits=fc_out, label=soft_labels)

        loss = link_prediction_loss

        if self._use_encoder:
            if self._use_local:
                loss = self._lambda * loss + self._beta * local_re_err
            if self._use_global:
                loss = self._lambda * loss + self._gamma * global_re_err

        return loss, fc_out

    def embedding(self, input_ids, node_type, input_mask):
        emb_out = self.node_embedding(input_ids)
        emb_out = torch.nn.Dropout(self._prepostprocess_dropout)(self.layer_norm1(emb_out))
        if self._use_node_heterogeneity:
            nodes_key = self.node_embedding_k(node_type)
            nodes_value = self.node_embedding_v(node_type)
            node_mask = torch.sign(node_type).unsqueeze(2)
            nodes_key = torch.mul(nodes_key, node_mask)
            nodes_value = torch.mul(nodes_value, node_mask)
        else:
            nodes_key = None
            nodes_value = None
        input_mask = input_mask.unsqueeze(2)
        self_attn_mask = torch.matmul(input_mask, input_mask.transpose(1, 2))
        self_attn_mask = 1000000.0 * (self_attn_mask - 1.0)
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)  # 1024x4个相同的11x64个mask

        return emb_out, nodes_key, nodes_value, n_head_self_attn_mask

    def decoder(self, _enc_out, e, incidence_matrix_T):
        # hye_enc_out = self.hyperedge_encoder(_enc_out, incidence_matrix_T.transpose(1, 2))
        hye_enc_out = e
        sigmoid = torch.nn.Sigmoid()
        re_incidence_matrix = sigmoid(_enc_out.bmm(hye_enc_out.transpose(-2, -1)))
        mseloss = torch.nn.MSELoss()
        re_err = mseloss(incidence_matrix_T.float().transpose(1, 2), re_incidence_matrix)
        return re_err

    def hyperedge_init(self, incidence_matrix, batch_size):
        if self._use_dynamic:
            mu = self.edges_mu.expand(incidence_matrix.size()[1], -1)
            sigma = self.edges_logsigma.exp().expand(incidence_matrix.size()[1], -1)
            e = mu + sigma * torch.randn((batch_size, mu.size(0), mu.size(1)), device=self._device)
        else:
            e = None
        return e


class SoftmaxWithCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, logits, label):
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -1.0 * torch.sum(torch.mul(label, logprobs), dim=1).squeeze()
        loss = torch.mean(loss)
        return loss




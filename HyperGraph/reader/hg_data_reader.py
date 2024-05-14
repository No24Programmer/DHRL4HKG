import collections
import json
import os.path
from random import sample
import numpy as np
import torch
from tqdm import tqdm


from HyperGraph.entity.HDataset import HDataset
from HyperGraph.entity.hfact_example import NaryExample
from HyperGraph.entity.hfact_teature import NaryFeature
from utils.file_rw import write_data_to_json
from utils.test_args import get_max_node


def read_examples(input_file, dataset=None, args=None):
    examples = []
    max_node_num = 0
    max_hyeredge = 0
    max_arity = 0
    with open(input_file, "r") as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            assert "N" in obj.keys() \
                   and "relation" in obj.keys() \
                   and "subject" in obj.keys() \
                   and "object" in obj.keys(), \
                "There are 4 mandatory fields: N, relation, subject, and object."
            arity = obj["N"]
            if args.use_100_data and arity > 2:
                continue
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            node_num = 3
            hyperedge_num = 0

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()

                for attribute in sorted(obj.keys()):
                    if attribute == "N" \
                            or attribute == "relation" \
                            or attribute == "subject" \
                            or attribute == "object":
                        continue

                    auxiliary_info[attribute] = sorted(obj[attribute])

                    node_num = node_num + 1 + len(obj[attribute])
                    hyperedge_num += 1
            else:
                hyperedge_num = 1
            max_node_num = node_num if node_num > max_node_num else max_node_num
            max_hyeredge = hyperedge_num if hyperedge_num > max_hyeredge else max_hyeredge
            max_arity = arity if arity > max_arity else max_arity

            example = NaryExample(
                arity=arity,  # arity,
                node_num=node_num,
                hyperedge_num=hyperedge_num,
                relation=relation,
                head=head,
                tail=tail,
                auxiliary_info=auxiliary_info)
            examples.append(example)

    return examples, max_node_num, max_hyeredge


def convert_examples_to_hypergraph(examples, vocabulary, max_node_num, max_hyperedge, dataset=None, args=None):

    neighbor_dict = get_neighbor_dict(examples, vocabulary)
    max_token = 0
    max_edge = 0
    one2many = 0
    features = []
    feature_id = 0
    for (example_id, example) in tqdm(enumerate(examples), total=len(examples)):
        use_for_one2many = False
        # get original input tokens and input mask
        rht = [example.relation, example.head, example.tail]
        rht_mask = [1, 1, 1]

        node_num = example.node_num  # HyperGraph node number
        hyperedge_num = example.hyperedge_num  # Hyperedge number
        incidence_matrix_T = []  # the hypergraph incidence matrix.

        orig_input_tokens = []
        orig_input_tokens.extend(rht)
        orig_input_mask = []
        orig_input_mask.extend(rht_mask)
        orig_type_label = [-1, 1, 1]  # relation/key = -1; entity/value = 1
        hyp_type_index = [-1, -1, -1]  # main triple = -1;
        node_type_label = [1, 2, 2]    # relation-1, head/tail entity-2, auxiliary relation-3, auxiliary entity-4, other-0

        node_id = 3
        if example.auxiliary_info is not None:
            for index, attribute in enumerate(example.auxiliary_info.keys()):
                hyperedge = [0 for i in range(max_node_num)]
                hyperedge[0] = 1  # relation
                hyperedge[1] = 1  # head
                hyperedge[2] = 1  # tail

                hyperedge[node_id] = 1  # attribute
                node_id += 1
                orig_input_tokens.append(attribute)
                orig_input_mask.append(1)
                orig_type_label.append(-1)
                hyp_type_index.append(index)
                node_type_label.append(3)

                for value in example.auxiliary_info[attribute]:
                    hyperedge[node_id] = 1  # value
                    node_id += 1
                    orig_input_tokens.append(value)
                    orig_input_mask.append(1)
                    orig_type_label.append(1)
                    hyp_type_index.append(index)
                    node_type_label.append(4)

                incidence_matrix_T.append(hyperedge)  # a hyperedge

                if args.test_one2many and len(example.auxiliary_info[attribute]) > 1:
                    use_for_one2many = True
                    one2many +=1

        else:
            incidence_matrix_T.append([1, 1, 1] + [0 for i in range(max_node_num-3)])
        assert node_id == example.node_num
        assert len(incidence_matrix_T) == example.hyperedge_num
        while len(incidence_matrix_T) < max_hyperedge:
            incidence_matrix_T.append([0 for i in range(max_node_num)])
        assert len(incidence_matrix_T) == max_hyperedge

        while len(orig_input_tokens) < max_node_num:
            orig_input_tokens.append("[PAD]")
            orig_input_mask.append(0)
            node_type_label.append(0)
        assert len(orig_input_tokens) == max_node_num

        if use_for_one2many:
            # generate a feature by masking each of the tokens
            for mask_position in range(max_node_num):
                if orig_input_tokens[mask_position] == "[PAD]":
                    continue
                mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
                mask_type = orig_type_label[mask_position]
                input_tokens = orig_input_tokens[:]
                input_tokens[mask_position] = "[MASK]"
                input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
                mask_edge_index = hyp_type_index[mask_position]
                node_type = node_type_label[:]
                assert len(input_tokens) == max_node_num
                assert len(input_ids) == max_node_num

                k = 10 if args.global_k is None else args.global_k
                max_node = get_max_node(dataset, k)
                neighbor_dict_list = get_sample_neighbor_list(example_id, example, neighbor_dict, k=k)
                g_input_ids, g_incidence_matrix, g_node_type, token, edge = load_neighbor_data(examples, vocabulary, input_ids,
                                                                                  incidence_matrix_T, node_type,
                                                                                  neighbor_dict_list,
                                                                                  max_node_num=max_node,
                                                                                  max_edge_num=k+1)
                max_token = token if max_token < token else max_token
                max_edge = edge if max_edge < edge else max_edge

                feature = NaryFeature(
                    feature_id=feature_id,
                    example_id=example_id,
                    input_tokens=input_tokens,
                    input_ids=input_ids,
                    input_mask=orig_input_mask,
                    mask_position=mask_position,
                    mask_label=mask_label,
                    mask_type=mask_type,
                    arity=example.arity,
                    incidence_matrix_T=incidence_matrix_T,
                    node_num=node_num,
                    hyperedge_num=hyperedge_num,
                    mask_edge_index=mask_edge_index,
                    node_type=node_type,
                    g_input_ids=g_input_ids,
                    g_incidence_matrix=g_incidence_matrix,
                    g_node_type=g_node_type
                )
                features.append(feature)
                feature_id += 1

    return features


def load_data(file_path, vocabulary, device, dataset=None, args=None):
    examples, max_node_num, max_hyeredge = read_examples(file_path, dataset, args)
    features = convert_examples_to_hypergraph(examples, vocabulary, max_node_num, max_hyeredge, dataset, args)

    input_ids = torch.tensor([f.input_ids for f in features]).to(device)
    input_mask = torch.FloatTensor([f.input_mask for f in features]).to(device)
    mask_position = torch.tensor([f.mask_position for f in features]).to(device)
    mask_label = torch.tensor([f.mask_label for f in features]).to(device)
    mask_type = torch.tensor([f.mask_type for f in features]).to(device)
    incidence_matrix_T = torch.tensor([f.incidence_matrix_T for f in features]).to(device)
    node_num = torch.tensor([f.node_num for f in features]).to(device)
    hyperedge_num = torch.tensor([f.hyperedge_num for f in features]).to(device)
    mask_edge_index = torch.tensor([f.mask_edge_index for f in features]).to(device)
    node_type = torch.tensor([f.node_type for f in features]).to(device)
    g_input_ids = torch.tensor([f.g_input_ids for f in features]).to(device)
    g_incidence_matrix = torch.tensor([f.g_incidence_matrix for f in features]).to(device)
    g_node_type = torch.tensor([f.g_node_type for f in features]).to(device)

    data_list = list()
    data_list.append(input_ids)
    data_list.append(input_mask)
    data_list.append(mask_position)
    data_list.append(mask_label)
    data_list.append(mask_type)
    data_list.append(incidence_matrix_T)
    data_list.append(node_num)
    data_list.append(hyperedge_num)
    data_list.append(mask_edge_index)
    data_list.append(node_type)
    data_list.append(g_input_ids)
    data_list.append(g_incidence_matrix)
    data_list.append(g_node_type)

    data_reader = HDataset(data_list)

    return data_reader



def generate_ground_truth(ground_truth_path, vocabulary, max_node_num, dataset=None, args=None):

    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    all_examples, _, m = read_examples(ground_truth_path, args=args)
    for (example_id, example) in enumerate(all_examples):
        # get padded input tokens and ids
        rht = [example.relation, example.head, example.tail]
        input_tokens = rht
        # aux_attributes = []
        # aux_values = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                # if dataset != "jf17k":
                input_tokens.append(attribute)
                for value in example.auxiliary_info[attribute]:
                    input_tokens.append(value)

        while len(input_tokens) < max_node_num:
            input_tokens.append("[PAD]")

        # input_tokens = rht + aux_attributes + aux_values
        input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
        assert len(input_tokens) == max_node_num
        assert len(input_ids) == max_node_num

        # get target answer for each pos and the corresponding key
        for pos in range(max_node_num):
            if input_ids[pos] == 0:
                continue
            key = " ".join([
                str(input_ids[x]) for x in range(max_node_num) if x != pos
            ])
            gt_dict[pos][key].append(input_ids[pos])

    return gt_dict




def get_neighbor_dict(examples, vocabulary):
    neighbor_dict = dict()
    for v in vocabulary.vocab:
        neighbor_dict[v] = []
    for i, example in enumerate(examples):
        head = example.head
        relation = example.relation
        tail = example.tail
        neighbor_dict[head].append(i)
        neighbor_dict[relation].append(i)
        neighbor_dict[tail].append(i)
        # if example.auxiliary_info is not None:
        #     for index, attribute in enumerate(example.auxiliary_info.keys()):
        #         neighbor_dict[attribute].append(i)
        #         for value in example.auxiliary_info[attribute]:
        #             neighbor_dict[value].append(i)
    return neighbor_dict


def get_sample_neighbor_list(example_id, example, neighbor_dict, k=0):
    head = example.head
    relation = example.relation
    tail = example.tail
    head_nei = neighbor_dict[head]
    relation_nei = neighbor_dict[relation]
    tail_nei = neighbor_dict[tail]
    h_r_nei = list(set(head_nei) & set(relation_nei))
    r_t_nei = list(set(relation_nei) & set(tail_nei))

    # a_nei = []

    # if example.auxiliary_info is not None:
    #     for index, attribute in enumerate(example.auxiliary_info.keys()):
    #         a_v = neighbor_dict[attribute]
    #         for value in example.auxiliary_info[attribute]:
    #             a_v = list(set(a_v) & set(neighbor_dict[value]))
    #             a_nei = a_nei + a_v

    neighbor = list((set(h_r_nei) | set(r_t_nei)) - {example_id})
    # neighbor = list((set(h_r_nei) | set(r_t_nei) | set(a_nei)) - {example_id})
    if k < len(neighbor):
        return sample(neighbor, k)
    else:
        return neighbor


def load_neighbor_data(examples, vocabulary, input_ids_, incidence_matrix_T_, node_type_, neighbor, max_node_num, max_edge_num):
    input_id = input_ids_.copy()
    incidence_m = incidence_matrix_T_.copy()
    node_t = node_type_.copy()

    incidence_matrix = np.array(incidence_m)
    incidence_matrix = np.where(np.sum(incidence_matrix, axis=0)[np.newaxis, :] != 0, 1, 0)
    edge_num, node_num = incidence_matrix.shape
    incidence_matrix = np.concatenate([incidence_matrix, np.zeros((edge_num, max_node_num-node_num))], axis=1)
    # node_type = node_type + ([0] * node_num * len(neighbor))
    # max_node_num = node_num * len(neighbor) + node_num
    # max_edge_num = edge_num * len(neighbor) + edge_num

    for n_i in neighbor:
        input_id, node_t, hyperedges = get_neighbor_feature(examples[n_i], input_id, node_t, vocabulary,
                                                                max_node_num)
        incidence_matrix = np.concatenate([incidence_matrix, np.array(hyperedges)], axis=0)

    max_token = len(input_id)
    max_edge = incidence_matrix.shape[0]
    while len(input_id) < max_node_num:
        input_id.append(0)
    while len(node_t) < max_node_num:
        node_t.append(0)
    if incidence_matrix.shape[0] < max_edge_num:
        incidence_matrix = np.concatenate(
            [incidence_matrix, np.zeros((max_edge_num - incidence_matrix.shape[0], max_node_num))], axis=0)

    return input_id, incidence_matrix, node_t, max_token, max_edge,


def get_neighbor_feature(example, input_ids, node_type, vocabulary, max_node_num):
    r_i = vocabulary.convert_token_to_id(example.relation)
    if r_i in input_ids:
        r_p = input_ids.index(r_i)
    else:
        r_p = len(input_ids)
        input_ids.append(r_i)
        node_type.append(1)

    h_i = vocabulary.convert_token_to_id(example.head)
    if h_i in input_ids:
        h_p = input_ids.index(h_i)
    else:
        h_p = len(input_ids)
        input_ids.append(h_i)
        node_type.append(2)

    t_i = vocabulary.convert_token_to_id(example.tail)
    if t_i in input_ids:
        t_p = input_ids.index(t_i)
    else:
        t_p = len(input_ids)
        input_ids.append(t_i)
        node_type.append(2)

    hyperedges = []
    hyperedge = np.zeros(max_node_num)
    hyperedge[r_p] = 1
    hyperedge[h_p] = 1
    hyperedge[t_p] = 1
    if example.auxiliary_info is not None:
        for index, attribute in enumerate(example.auxiliary_info.keys()):
            a_i = vocabulary.convert_token_to_id(attribute)
            if a_i in input_ids:
                a_p = input_ids.index(a_i)
            else:
                a_p = len(input_ids)
                input_ids.append(a_i)
                node_type.append(3)
            hyperedge[a_p] = 1

            for value in example.auxiliary_info[attribute]:
                v_i = vocabulary.convert_token_to_id(value)
                if v_i in input_ids:
                    v_p = input_ids.index(v_i)
                else:
                    v_p = len(input_ids)
                    input_ids.append(v_i)
                    node_type.append(4)
                hyperedge[v_p] = 1
        hyperedges.append(hyperedge)
    else:
        hyperedge = np.zeros(max_node_num)
        hyperedge[r_p] = 1
        hyperedge[h_p] = 1
        hyperedge[t_p] = 1
        hyperedges.append(hyperedge)
    return input_ids, node_type, hyperedges


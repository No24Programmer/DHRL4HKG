

def get_max_node(dataset, k=10):
    if k == 10:
        max_node_dict = {
            "jf17k": 55,
            "wikipeople": 70,
            "wd50k": 120,
            "wd50k_33_raw": 150,
            "wd50k_66_raw": 180,
            "wd50k_100_raw": 190,
        }
    elif k == 1:
        max_node_dict = {
            "jf17k": 35,
            "wikipeople": 30,
            "wd50k": 35,
        }
    elif k == 5:
        max_node_dict = {
            "jf17k": 35,
            "wikipeople": 45,
            "wd50k": 90,
        }
    elif k == 15:
        max_node_dict = {
            "jf17k": 65,
            "wikipeople": 90,
            "wd50k": 130,
        }
    elif k == 20:
        max_node_dict = {
            "jf17k": 75,
            "wikipeople": 100,
            "wd50k": 135,
        }

    max_node = 0
    if dataset in max_node_dict.keys():
        max_node = max_node_dict[dataset]
    return max_node


def set_default_args(args, dataset_name):
    # Set data paths, vocab paths and data processing options.
    args.train_file = "./data/{}/train.json".format(dataset_name)
    args.valid_file = "./data/{}/valid.json".format(dataset_name)
    args.test_file = "./data/{}/test.json".format(dataset_name)

    args.predict_file = "./data/{}/test.json".format(dataset_name)
    args.ground_truth_path = "./data/{}/all.json".format(dataset_name)
    args.vocab_path = "./data/{}/vocab.txt".format(dataset_name)

    if dataset_name == "wikipeople":
        args.batch_size = 1024
        args.vocab_size = 47960
        args.num_relations = 193
        args.max_seq_len = 15   # 17
        args.max_arity = 9
        args.max_hyperedge_num = 6

    elif dataset_name == "jf17k":
        # args.batch_size = 1024
        args.vocab_size = 29148
        args.num_relations = 501
        args.max_seq_len = 11
        args.max_arity = 6
        args.max_hyperedge_num = 4
        args.epoch = 200

    elif dataset_name == "wd50k":
        args.batch_size = 256
        args.vocab_size = 47688
        args.num_relations = 531
        args.max_seq_len = 18  # 63
        args.max_arity = 32
        args.max_hyperedge_num = 5

    elif dataset_name == "wd50k_33_raw":
        args.vocab_size = 38599
        args.num_relations = 474
        args.max_seq_len = 34
        args.max_arity = 32
        args.max_hyperedge_num = 5
        args.epoch = 100

    elif dataset_name == "wd50k_66_raw":
        args.vocab_size = 27751
        args.num_relations = 403
        args.max_seq_len = 69
        args.max_arity = 67
        args.max_hyperedge_num = 5
        args.epoch = 100

    elif dataset_name == "wd50k_100_raw":
        args.vocab_size = 19071
        args.num_relations = 278
        args.max_seq_len = 69
        args.max_arity = 67
        args.max_hyperedge_num = 5
        args.epoch = 100

    return args


import logging
import argparse
import os
import time
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from reader.vocab_reader import Vocabulary
from utils.args import ArgumentGroup
from HyperGraph.reader.hg_data_reader import *
from utils.evaluation import predict
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.train_args import set_default_args
from HyperGraph.model.DHE import HypaerGraph_Model

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

# =====================  Argument Parser  =============================== #
parser = argparse.ArgumentParser()

data_g = ArgumentGroup(parser, "data",     "Data paths, vocab paths and data processing options.")
data_g.add_arg("dataset_name",            str,    "wikipeople",   "Dataset name")
data_g.add_arg("train_file",              str,    None,      "Data for training.")
data_g.add_arg("valid_file",              str,    None,      "Data for valid.")
data_g.add_arg("test_file",               str,    None,      "Data for test.")
data_g.add_arg("predict_file",            str,    None,      "Data for prediction.")
data_g.add_arg("ground_truth_path",       str,    None,      "Path to ground truth.")
data_g.add_arg("vocab_path",              str,    None,      "Path to vocabulary.")
data_g.add_arg("vocab_size",              int,    None,      "Size of vocabulary.")
data_g.add_arg("num_relations",           int,    None,      "Number of relations.")
data_g.add_arg("max_seq_len",             int,    None,      "Max sequence length.")
data_g.add_arg("max_arity",               int,    None,      "Max arity.")
data_g.add_arg("entity_soft_label",       float,  0.1,       "Label smoothing rate for masked entities.")
data_g.add_arg("relation_soft_label",     float,  0.1,       "Label smoothing rate for masked relations.")


model_g = ArgumentGroup(parser, "model",    "model and checkpoint configuration.")
model_g.add_arg("num_hidden_layers",       int,    6,        "Number of hidden layers.")
model_g.add_arg("num_attention_heads",     int,    4,        "Number of attention heads.")
model_g.add_arg("hidden_size",             int,    512,      "Hidden size.")
model_g.add_arg("intermediate_size",       int,    1024,     "Intermediate size.")
model_g.add_arg("hidden_act",              str,    "gelu",   "Hidden act.")
model_g.add_arg("hidden_dropout_prob",     float,  0.1,      "Hidden dropout ratio.")
model_g.add_arg("attention_dropout_prob",  float,  0.1,      "Attention dropout ratio.")
model_g.add_arg("num_node_type",           int,    6,        "Number of node type for heterogeneity")
model_g.add_arg("use_decoder",             bool,   True,     "use Hypergraph restructure decoder")
model_g.add_arg("use_dynamic",             bool,   True,     "use hypergraph init in dynamic hypergraph")
model_g.add_arg("use_global",              bool,   True,     "use global view")
model_g.add_arg("use_local",               bool,   True,     "use local view")
model_g.add_arg("use_node_heterogeneity",  bool,   True,     "use node heterogeneity")
model_g.add_arg("global_k",                int,    10,        "random sample k neighbor")
model_g.add_arg("lambda",                  float,  1.0,       "Objective function parameter lambda of link prediction.")
model_g.add_arg("beta",                    float,  1.0,       "Objective function parameter beta of local view.")
model_g.add_arg("gamma",                   float,  1.0,       "Objective function parameter gamma of global view.")
model_g.add_arg("checkpoints",             str,    "./logs/checkpoints",   "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("batch_size",        int,    1024,                   "Batch size.")
train_g.add_arg("epoch",             int,    200,                    "Number of training epochs.")
train_g.add_arg("learning_rate",     float,  3e-4,                   "Learning rate with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",  "scheduler of learning rate.",
                choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("warmup_proportion", float,  0.1,                    "Proportion of training steps for lr warmup.")
train_g.add_arg("weight_decay",      float,  0.01,                   "Weight decay rate for L2 regularizer.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    1000,    "Step intervals to print loss.")
log_g.add_arg("verbose",             bool,   False,   "Whether to output verbose log.")
log_g.add_arg("logs_save_dir",       str,    "./logs",      "Path to save logs.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,                "If set, use GPU for training.")
run_type_g.add_arg("device",                       str,    "0",            "{0123}^n,1<=n<=4,the first cuda is used as master device and others are used for data parallel")
run_type_g.add_arg("use_checkpoints",              bool,   True,               "")
run_type_g.add_arg("checkpoints_path",             str,    "./logs/checkpoints/DHE_model_20240404-164020.tar", "")
run_type_g.add_arg("model_type",                   str,    "DHE",    "")
run_type_g.add_arg("iteration",                    int,    2,              "")
run_type_g.add_arg("save_model",                   bool,   True,           "")


args = parser.parse_args()
# =====================  Argument Parser END  =============================== #


def develepment_train(args, time_):
    args = set_default_args(args, args.dataset_name)
    config = vars(args)
    if args.use_cuda:
        device = torch.device(f"cuda:{args.device[0]}")
        devices = []
        for i in range(len(args.device)):
            devices.append(torch.device(f"cuda:{args.device[i]}"))
    else:
        device = torch.device("cpu")
        config["device"] = "cpu"
    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    # ************* load dataset ***************
    vocabulary = Vocabulary(args.vocab_path, args.num_relations, args.vocab_size - args.num_relations - 2)

    logger.info("loading train dataloader")
    train_data_reader = load_data(args.train_file, vocabulary, device, dataset=args.dataset_name, args=args)
    train_pyreader = DataLoader(train_data_reader, batch_size=args.batch_size, shuffle=True,
                                           drop_last=False)

    logger.info("loading valid dataloader")
    valid_data_reader = load_data(args.valid_file, vocabulary, device, dataset=args.dataset_name, args=args)
    valid_pyreader = DataLoader(valid_data_reader, batch_size=args.batch_size, shuffle=False,
                                           drop_last=False)

    all_facts = generate_ground_truth(args.ground_truth_path, vocabulary, args.max_seq_len, dataset=args.dataset_name, args=args)

    # ************* Create model ****************
    if len(devices) > 1:
        model = torch.nn.DataParallel(HypaerGraph_Model(config), device_ids=devices)
        model.to(device)
    else:
        model = HypaerGraph_Model(config).to(device)
    logger.info(model)

    t_total = len(train_pyreader) * args.epoch
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    iteration_start = 1

    if args.use_checkpoints:
        logger.info("loal checkpoint model for train")
        checkpoint = torch.load(os.path.join(args.checkpoints, "{}_model.tar".format(args.model_type)))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration_start = checkpoint['epoch'] + 1

    # ************* Training ********************
    max_entity_mmr = 0
    for iteration in range(iteration_start, args.epoch):
        logger.info("iteration " + str(iteration))
        t1_strat = time.time()
        # --------- train -----------
        for j, data in tqdm(enumerate(train_pyreader), total=len(train_pyreader)):
            model.train()
            optimizer.zero_grad()
            # Hypergraph_001
            # loss = model(data)
            loss, fc_out = model(data, is_train=True)
            if len(devices) > 1:
                loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if j % 100 == 0:
                # logger.info(str(j) + ' , loss: ' + str([loss[i].item() for i in range(len(loss))]))
                logger.info(str(j) + ' , loss: ' + str(loss.item()))
        # --------- validation -----------
        if iteration % args.iteration == 0:
            logger.info("Train time = {:.3f} s".format(round(time.time() - t1_strat, 2)))
            # Start validation and testing

            logger.info("Start validation")
            model.eval()
            with torch.no_grad():
                entity_mmr = predict(model, valid_pyreader, all_facts, None, None, device, logger, args.model_type)

            t1_end = time.time()
            t1 = round(t1_end - t1_strat, 2)
            logger.info("Iteration time = {:.3f} s".format(t1))

            # save model
            if entity_mmr > max_entity_mmr:
                max_entity_mmr = entity_mmr
                logger.info("=============== Best performance ============== ")
                if args.save_model:
                    torch.save({
                        'epoch': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(args.checkpoints, "{}_model_{}.tar".format(args.model_type, time_)))
                    logger.info("save model to {}".format(
                        os.path.join(args.checkpoints, "{}_model_{}.tar".format(args.model_type, time_))))


if __name__ == '__main__':
    # log file
    data_file_path = str(time.strftime("%Y-%m-%d", time.localtime()))
    logs_save_dir = os.path.join(args.logs_save_dir, args.dataset_name, data_file_path)
    if not os.path.exists(logs_save_dir):
        os.makedirs(logs_save_dir)
    time_ = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    fileHandler = logging.FileHandler(
        os.path.join(logs_save_dir, '{}_{}_{}.log'.format(time_, args.dataset_name, args.device)))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    develepment_train(args, time_)





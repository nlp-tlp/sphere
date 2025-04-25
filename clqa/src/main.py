#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from model.gqe import GQE
from model.query2box import Query2Box
from model.sphere import SpherE
from tensorboardX import SummaryWriter
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import flatten_query, parse_time, set_global_seed, eval_tuple, display_memory_usage
#  import shutil

from train import train_step, test_step
#  import statsmodels.api as sm

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
# ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']
all_tasks = list(name_query_dict.keys())


def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', default=None, type=str,
                        help="KG data path")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--prefix', default=None, type=str,
                        help='prefix of the log path')
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--select_train_query', default=None, type=str,
                        help="select one or several training query structures, 1p, 1p.2p.3p, etc.")
    parser.add_argument('--select_test_query', default=None, type=str,
                        help="select one or several test query structures, 1p, 1p.2p.3p, etc.")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-n', '--negative_sample_size', default=128,
                        type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500,
                        type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=24.0,
                        type=float, help="margin in the loss")
    parser.add_argument('-mw', '--modulus_weight', default=1.0, type=float)
    parser.add_argument('-pw', '--phase_weight', default=0.5, type=float)
    parser.add_argument('-pd', '--probability_distribution',
                        action='store_true', help="activate probability distribution feature for weighting self-adversarial negative sampling loss")
    parser.add_argument('--loss_type', default="NS", type=str, choices=['NS', 'CE'],
                        help='loss type, negative sampling loss or cross entropy loss')
    parser.add_argument('-a', '--adversarial_temperature',
                        default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024,
                        type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1,
                        type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001,
                        type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10,
                        type=int, help="used to speed up torch.dataloader")
    parser.add_argument('--max_steps', default=100000,
                        type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None,
                        type=int, help="no need to set manually, will configure automatically")

    parser.add_argument('--nentity', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0,
                        help='DO NOT MANUALLY SET')

    parser.add_argument('--model', default='sphere', type=str, choices=['noname', 'vec', 'box', 'sphere'],
                        help='list of model names')
    parser.add_argument('-cenr', '--center_mode', default="(none,0.02)", type=str,
                        help='(offset activation, center_mode), center_mode balances the inside dist and outside dist')

    parser.add_argument('--checkpoint_path', default=None,
                        type=str, help='path for loading the checkpoints')
    parser.add_argument('--save_checkpoint_steps', default=25000,
                        type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000,
                        type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100,
                        type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000,
                        type=int, help='valid/test log every xx steps')

    return parser.parse_args(args)


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %.3f' %
                     (mode, metric, step, metrics[metric]))


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = test_step(
        model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(
            mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join(
                [mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join(
                [query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar(
            "_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)

    return all_metrics


def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    train_queries = pickle.load(
        open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(
        open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(
        open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(
        open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(
        open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(
        open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(
        open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(
        open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([
                name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return train_queries, train_answers, \
        valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers, \



def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print(f"overwritting args.save_path")
    args.save_path = os.path.join(
        #  prefix, args.data_path.split('/')[-1], args.tasks, args.model)
        prefix, args.data_path.split('/')[-1], args.model)

    if args.model in ['vec', 'box', 'sphere']:
        tmp_str = "g-{}-dim-{}".format(args.gamma, args.hidden_dim)

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"logging to {args.save_path}")
    if not args.do_train:  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    #  shutil.copy("clqa/src/model/sphere.py", args.save_path)
    #  shutil.copy("clqa/src/main.py", args.save_path)
    #  print(f"copying model and main {args.save_path}")

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('-------------------------------'*3)
    logging.info('Model Name: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, \
        valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers = load_data(
            args, tasks)

    ################################################################################################
    # select specific train, valid, test queries
    if args.select_train_query != None:
        lst_query_structure = args.select_train_query.split(".")  # ["1p"]
        select_query_structure = [name_query_dict[key]
                                  for key in lst_query_structure]
        train_queries = dict(
            (key, train_queries[key]) for key in select_query_structure if key in train_queries)
    else:
        # use default settings
        pass

    if args.select_test_query != None:
        lst_query_structure = args.select_test_query.split(".")
        select_query_structure = [name_query_dict[key]
                                  for key in lst_query_structure]
        valid_queries = dict(
            (key, valid_queries[key]) for key in select_query_structure if key in valid_queries)
        test_queries = dict(
            (key, test_queries[key]) for key in select_query_structure if key in test_queries)
    else:
        # use default settings
        pass
    ################################################################################################

    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries:
            logging.info(
                query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, nentity, nrelation,
                         args.negative_sample_size, train_answers,
                         args.loss_type),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, nentity, nrelation,
                             args.negative_sample_size, train_answers,
                             args.loss_type),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(
                query_name_dict[query_structure]+": "+str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(
                query_name_dict[query_structure]+": "+str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries[:],
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )
    if args.model == "vec":
        model = GQE(
            nentity=nentity,
            nrelation=nrelation,
            emb_dim=args.hidden_dim,
            gamma=args.gamma,
            model_name=args.model,
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda,
            query_name_dict=query_name_dict
        )
    elif args.model == "box":
        model = Query2Box(
            nentity=nentity,
            nrelation=nrelation,
            emb_dim=args.hidden_dim,
            gamma=args.gamma,
            model_name=args.model,
            center_mode=eval_tuple(args.center_mode),
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda,
            query_name_dict=query_name_dict
        )
    elif args.model == "sphere":
        model = SpherE(
            nentity=nentity,
            nrelation=nrelation,
            emb_dim=args.hidden_dim,
            gamma=args.gamma,
            model_name=args.model,
            center_mode=eval_tuple(args.center_mode),
            mod_weight=args.modulus_weight,
            phase_weight=args.phase_weight,
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda,
            query_name_dict=query_name_dict
        )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate,
        )
        warm_up_steps = args.max_steps // 2

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(
            args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step
    if args.model in ['box', 'sphere']:
        logging.info(
            f"{args.model} mode (inside_dis_lambda) = {args.center_mode}")

    logging.info('tasks = %s' % args.tasks)
    logging.info('(optional) select train query = %s' %
                 args.select_train_query)
    logging.info('(optional) select test query = %s' % args.select_test_query)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %.5f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('loss type = %s' % args.loss_type)
    if args.probability_distribution:
        logging.info(
            'activate probability distribution for weighting self-adversarial negative sampling loss')
    if args.model in ['sphere']:
        logging.info(f"modulus weight: {args.modulus_weight}")
        logging.info(f"phase weight: {args.phase_weight}")
    logging.info('seed = %d' % args.seed)

    if args.do_train:
        training_logs = []
        # #Training Loop
        for step in tqdm(range(init_step, args.max_steps)):
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            log = train_step(
                model, optimizer, train_path_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_'+metric, log[metric], step)
            if train_other_iterator is not None:
                log = train_step(
                    model, optimizer, train_other_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                log = train_step(
                    model, optimizer, train_path_iterator, args, step)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' %
                             (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(
                        model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(
                        model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum(
                        [log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []
                #  display_memory_usage() # show gpu memory

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)

    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers,
                                    args, test_dataloader, query_name_dict, 'Test', step, writer)

    logging.info(f"\n")
    logging.info("Training finished!!\n")


if __name__ == '__main__':
    main(parse_args())

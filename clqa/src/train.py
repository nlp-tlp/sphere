#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import collections
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F


def train_step(model, optimizer, train_iterator, args, step):
    model.train()
    optimizer.zero_grad()
    positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(
        train_iterator)

    batch_queries_dict = collections.defaultdict(list)
    batch_idxs_dict = collections.defaultdict(list)
    # group queries with same structure
    for i, query in enumerate(batch_queries):
        batch_queries_dict[query_structures[i]].append(query)
        batch_idxs_dict[query_structures[i]].append(i)
    for query_structure in batch_queries_dict:
        if args.cuda:
            batch_queries_dict[query_structure] = torch.LongTensor(
                batch_queries_dict[query_structure]).cuda()
        else:
            batch_queries_dict[query_structure] = torch.LongTensor(
                batch_queries_dict[query_structure])
    if args.cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

    positive_sample_label = positive_sample
    if args.loss_type == "CE":
        positive_sample = None

    positive_logit, negative_logit, subsampling_weight, all_idxs, _, _, _, _ = \
        model(positive_sample, negative_sample, subsampling_weight,
              batch_queries_dict, batch_idxs_dict)

    # cross entropy loss
    if args.loss_type == "CE":
        labels = positive_sample_label[all_idxs]
        loss = nn.CrossEntropyLoss(reduction='mean')(negative_logit, labels)
        positive_sample_loss = torch.zeros(1)
        negative_sample_loss = torch.zeros(1)
    # negative sampling loss
    else:
        if args.probability_distribution:
            negative_score = (F.softmax(negative_logit * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_logit)).sum(dim=1)
        else:
            # default
            negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

    loss.backward()
    optimizer.step()
    log = {
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item(),
    }
    return log


def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
    model.eval()

    step = 0
    total_steps = len(test_dataloader)
    logs = collections.defaultdict(list)

    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
            batch_queries_dict = collections.defaultdict(list)
            batch_idxs_dict = collections.defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                if args.cuda:
                    batch_queries_dict[query_structure] = torch.LongTensor(
                        batch_queries_dict[query_structure]).cuda()
                else:
                    batch_queries_dict[query_structure] = torch.LongTensor(
                        batch_queries_dict[query_structure])
            if args.cuda:
                negative_sample = negative_sample.cuda()

            #  cardinality
            #  _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            _, negative_logit, _, idxs, _, _, _, _ = model(
                None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
            if len(argsort) == args.test_batch_size:
                # achieve the ranking of all entities
                ranking = ranking.scatter_(
                    1, argsort, model.batch_entity_range)
            else:  # otherwise, create a new torch Tensor for batch_entity_range
                if args.cuda:
                    ranking = ranking.scatter_(1,
                                               argsort,
                                               torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                  1).cuda()
                                               )  # achieve the ranking of all entities
                else:
                    ranking = ranking.scatter_(1,
                                               argsort,
                                               torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                  1)
                                               )  # achieve the ranking of all entities
            for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                hard_answer = hard_answers[query]
                easy_answer = easy_answers[query]
                num_hard = len(hard_answer)
                num_easy = len(easy_answer)
                assert len(hard_answer.intersection(easy_answer)) == 0
                cur_ranking = ranking[idx, list(
                    easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy
                if args.cuda:
                    answer_list = torch.arange(
                        num_hard + num_easy).to(torch.float).cuda()
                else:
                    answer_list = torch.arange(
                        num_hard + num_easy).to(torch.float)
                cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                # only take indices that belong to the hard answers
                cur_ranking = cur_ranking[masks]

                mrr = torch.mean(1./cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean(
                    (cur_ranking <= 10).to(torch.float)).item()

                logs[query_structure].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                    'num_hard_answer': num_hard,
                })

            if step % args.test_log_steps == 0:
                logging.info('Evaluating the model... (%d/%d)' %
                             (step, total_steps))

            step += 1

    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in logs:
        for metric in logs[query_structure][0].keys():
            if metric in ['num_hard_answer']:
                continue
            metrics[query_structure][metric] = sum(
                [log[metric] for log in logs[query_structure]])/len(logs[query_structure])
        metrics[query_structure]['num_queries'] = len(
            logs[query_structure])

    return metrics

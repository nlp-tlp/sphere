#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import mod


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


def Identity(x):
    return x


class SphereProjection(nn.Module):
    def __init__(self, dim, embedding_range, pi):
        super(SphereProjection, self).__init__()

    def forward(self, mod_embedding, phase_embedding, radius_embedding,
                r_mod_embedding, r_phase_embedding, r_bias_embedding, r_radius_embedding):

        r_bias_embedding = torch.clamp(r_bias_embedding, max=1)
        mask = (r_bias_embedding < -r_mod_embedding)
        r_bias_embedding[mask] = -r_mod_embedding[mask]

        mod_embedding = mod_embedding * (r_mod_embedding + r_bias_embedding)
        #  mod_embedding = mod_embedding * r_mod_embedding

        phase_embedding = phase_embedding + r_phase_embedding

        radius_embedding = radius_embedding * r_radius_embedding

        return mod_embedding, phase_embedding, \
            r_bias_embedding, radius_embedding


class SpherE(nn.Module):
    def __init__(self, nentity, nrelation, emb_dim, gamma,
                 model_name, center_mode=None,
                 mod_weight=1.0, phase_weight=0.5,
                 test_batch_size=1, use_cuda=False,
                 query_name_dict=None, is_emb_e=False, e_idx=0):
        super(SpherE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.epsilon = 2.0
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda(
        ) if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict

        self.entity_dim = emb_dim
        self.relation_dim = emb_dim

        self.activation, self.cen = center_mode

        self.mod_weight, self.phase_weight = mod_weight, phase_weight
        self.pi = 3.14159262358979323846

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / emb_dim]),
            requires_grad=False
        )

        self.entity_mod_embedding = nn.Parameter(
            torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_mod_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.entity_phase_embedding = nn.Parameter(
            torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_phase_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.entity_radius_embedding = nn.Parameter(
            torch.zeros(nentity, 1))
        nn.init.uniform_(
            tensor=self.entity_radius_embedding,
            a=0.,
            b=self.embedding_range.item()
        )

        self.relation_mod_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim))
        nn.init.ones_(tensor=self.relation_mod_embedding)

        self.relation_phase_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_phase_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_bias_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim))
        nn.init.zeros_(tensor=self.relation_bias_embedding)

        self.relation_radius_embedding = nn.Parameter(
            torch.zeros(nrelation, 1))
        nn.init.ones_(tensor=self.relation_radius_embedding)

        self.phase_weight = nn.Parameter(torch.Tensor(
            [[self.phase_weight * self.embedding_range.item()]]))
        self.mod_weight = nn.Parameter(torch.Tensor([[self.mod_weight]]))

        self.projection_net = SphereProjection(
            self.entity_dim, self.embedding_range, self.pi)

    def convert_radius(self, x, activation):
        if activation == "absolute":
            return torch.abs(x)
        elif activation == "none":
            return torch.abs(x)
        else:
            return x

    def forward(self, positive_sample, negative_sample, subsampling_weight,
                batch_queries_dict, batch_idxs_dict):
        return self.forward_sphere(positive_sample, negative_sample, subsampling_weight,
                                   batch_queries_dict, batch_idxs_dict)

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using SpherE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        # whether the current query tree has merged to one branch and only need to do relation traversal,
        # e.g., path queries
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                e_idx = queries[:, idx]
                mod_embedding = torch.index_select(
                    self.entity_mod_embedding, dim=0, index=e_idx)

                phase_embedding = torch.index_select(
                    self.entity_phase_embedding, dim=0, index=e_idx)
                phase_embedding = phase_embedding / \
                    (self.embedding_range.item() / self.pi)

                radius_embedding = self.convert_radius(torch.index_select(
                    self.entity_radius_embedding, dim=0, index=e_idx) / self.embedding_range.item(), self.activation)
                idx += 1
            else:
                mod_embedding, phase_embedding, r_bias_embedding, radius_embedding, idx = self.embed_query(
                    queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "negation is N/A at this step"
                else:
                    r_idx = queries[:, idx]
                    r_mod_embedding = torch.abs(torch.index_select(
                        self.relation_mod_embedding, dim=0, index=r_idx))

                    r_phase_embedding = torch.index_select(
                        self.relation_phase_embedding, dim=0, index=r_idx)
                    r_phase_embedding = r_phase_embedding / \
                        (self.embedding_range.item() / self.pi)

                    r_bias_embedding = torch.index_select(
                        self.relation_bias_embedding, dim=0, index=r_idx)

                    r_radius_embedding = torch.abs(torch.index_select(
                        self.relation_radius_embedding, dim=0, index=r_idx))

                    mod_embedding, phase_embedding, r_bias_embedding, radius_embedding = self.projection_net(
                        mod_embedding, phase_embedding, radius_embedding,
                        r_mod_embedding, r_phase_embedding, r_bias_embedding, r_radius_embedding)
                idx += 1
        else:
            assert False, "intersection is N/A at this step"

        return mod_embedding, phase_embedding, r_bias_embedding, radius_embedding, idx

    def get_embedding_e(self, e_idx):

        # mod
        mod_embedding = torch.index_select(
            self.entity_mod_embedding, dim=0, index=e_idx)

        # phase
        phase_embedding = torch.index_select(
            self.entity_phase_embedding, dim=0, index=e_idx)
        phase_embedding = phase_embedding / \
            (self.embedding_range.item() / self.pi)

        # radius
        radius_embedding = torch.index_select(
            self.entity_radius_embedding, dim=0, index=e_idx)
        radius_embedding = self.convert_radius(
            radius_embedding / self.embedding_range.item(), self.activation)

        return mod_embedding, phase_embedding, radius_embedding

    def cal_logit_sphere(self, mod_embedding, phase_embedding, r_bias_embedding, radius_embedding,
                         tail_mod_embedding, tail_phase_embedding, tail_radius_embedding):

        mod_dist = mod_embedding - tail_mod_embedding * (1 - r_bias_embedding)
        mod_dist = torch.norm(mod_dist, dim=-1)

        phase_dist = phase_embedding - tail_phase_embedding
        phase_dist = torch.sum(torch.abs(torch.sin(phase_dist / 2)), dim=-1)

        radius_dist = radius_embedding + tail_radius_embedding
        radius_dist = torch.sum(torch.abs(radius_dist), dim=-1)

        dist = self.mod_weight * mod_dist + self.phase_weight * phase_dist - \
            self.cen * radius_dist

        return self.gamma - dist

    def forward_sphere(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_mod_embeddings, all_phase_embeddings, all_r_bias_embeddings, all_radius_embeddings, all_idxs = [], [], [], [], []
        for query_structure in batch_queries_dict:
            mod_embedding, phase_embedding, r_bias_embedding, radius_embedding, _ = self.embed_query(
                batch_queries_dict[query_structure], query_structure, 0)
            all_mod_embeddings.append(mod_embedding)
            all_phase_embeddings.append(phase_embedding)
            all_r_bias_embeddings.append(r_bias_embedding)
            all_radius_embeddings.append(radius_embedding)
            all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_mod_embeddings) > 0 and len(all_phase_embeddings) > 0 and len(all_radius_embeddings) > 0:
            all_mod_embeddings = torch.cat(
                all_mod_embeddings, dim=0).unsqueeze(1)
            all_phase_embeddings = torch.cat(
                all_phase_embeddings, dim=0).unsqueeze(1)
            all_r_bias_embeddings = torch.cat(
                all_r_bias_embeddings, dim=0).unsqueeze(1)
            all_radius_embeddings = torch.cat(
                all_radius_embeddings, dim=0).unsqueeze(1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs]

        if type(positive_sample) != type(None):
            if len(all_mod_embeddings) > 0:
                pos_idx = positive_sample[all_idxs]
                tail_mod_embedding = torch.index_select(
                    self.entity_mod_embedding, dim=0, index=pos_idx).unsqueeze(1)

                tail_phase_embedding = torch.index_select(
                    self.entity_phase_embedding, dim=0, index=pos_idx).unsqueeze(1)
                tail_phase_embedding = tail_phase_embedding / \
                    (self.embedding_range.item() / self.pi)

                tail_radius_embedding = self.convert_radius(torch.index_select(
                    self.entity_radius_embedding, dim=0, index=pos_idx).unsqueeze(1) / self.embedding_range.item(), self.activation)
                positive_logit = self.cal_logit_sphere(all_mod_embeddings, all_phase_embeddings,
                                                       all_r_bias_embeddings, all_radius_embeddings,
                                                       tail_mod_embedding, tail_phase_embedding, tail_radius_embedding)
            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_mod_embedding.device)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_mod_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape

                tail_mod_embedding_neg = torch.index_select(self.entity_mod_embedding, dim=0,
                                                            index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)

                tail_phase_embedding_neg = torch.index_select(self.entity_phase_embedding, dim=0,
                                                              index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                tail_phase_embedding_neg = tail_phase_embedding_neg / \
                    (self.embedding_range.item() / self.pi)

                tail_radius_embedding_neg = self.convert_radius(torch.index_select(self.entity_radius_embedding, dim=0,
                                                                                   index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1) / self.embedding_range.item(), self.activation)

                negative_logit = self.cal_logit_sphere(all_mod_embeddings, all_phase_embeddings,
                                                       all_r_bias_embeddings, all_radius_embeddings,
                                                       tail_mod_embedding_neg, tail_phase_embedding_neg,
                                                       tail_radius_embedding_neg)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_mod_embedding.device)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs, \
            all_mod_embeddings, all_phase_embeddings, all_radius_embeddings, all_r_bias_embeddings

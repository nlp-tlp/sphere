from clqa.src.model.gqe import GQE
from clqa.src.model.query2box import Query2Box
from clqa.src.model.sphere import SpherE
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
import os
import sys

from clqa.src.util import eval_tuple, parse_time, display_memory_usage

sys.path.append("rp")

pi = 3.14159265358979323846


@torch.no_grad()
def clqa_forward(model, h, r, batch_idxs_dict, negative_sample):
    """
    Forward to models 
    """

    ent_sample = h.size(0)
    r = r.unsqueeze(-1).repeat(ent_sample, 1)  # shape (ent_sample, 1)
    h = h.unsqueeze(-1)  # shape (ent_sample, 1)
    positive_sample = torch.cat([h, r], dim=-1)  # shape (ent_sample, 2)

    batch_queries_dict = {("e", ("r",)): positive_sample}
    model.eval()

    _, negative_logit, _, _, _, _, _, _ = model(
        None, negative_sample, None, batch_queries_dict, batch_idxs_dict
    )

    return negative_logit  # (ent_sample, nentity)


@torch.no_grad()
def neural_adj_matrix(
    model,
    pre_model,
    rel,
    nentity,
    device,
    thrshd,
    adj_list_i,
    ent_sample,
    batch_idxs_dict,
    batch_idxs_dict_last,
    negative_sample,
):
    softmax = nn.Softmax(dim=1)
    relation_embedding = torch.zeros(nentity, nentity).to(torch.float)
    r = torch.LongTensor([rel]).to(device)

    # count number of connected tails for available heads
    num = torch.zeros(nentity, 1).to(torch.float).to(device)
    for h, t in adj_list_i:
        num[h, 0] += 1
    # num[h, 0] can be greater than one for some heads
    num = torch.maximum(num, torch.ones(nentity, 1).to(torch.float).to(device))

    for i in range(0, nentity, ent_sample):
        step = min(nentity, i + ent_sample)
        # h is a tensor [0-99], [100-199], ..., [14400-14499], [14500-14505]
        h = torch.arange(i, step).to(device)

        if i == 0:
            display_memory_usage()

        if pre_model == "clqa":
            if h.size(0) < ent_sample:
                batch_idxs_dict = batch_idxs_dict_last
            score = clqa_forward(
                model, h, r, device, nentity, batch_idxs_dict, negative_sample
            )

        # calibration function - step 1
        score = softmax(score) * num[i:step, :]
        # num[i:step, :] shape:                     # (ent_sample, 1)
        # normalized_score, score shape:            # (ent_sample, nentity)

        # space complextity: filter only scores higher than threshold, set the rest to zero
        mask = (score >= thrshd).to(torch.float)
        score = score * mask
        relation_embedding[i:step, :] = score.to("cpu")

    # calibration function - step 2
    # round the r(e_j, e_i), set delta 0.0001
    delta = 0.0001
    relation_embedding = (relation_embedding >= 1).to(torch.float) * (1 - delta) + (
        relation_embedding < 1
    ).to(torch.float) * relation_embedding
    for h, t in adj_list_i:
        relation_embedding[h, t] = 1.0

    return relation_embedding


def load_clqa(
    model_name,
    model_path,
    device,
    nentity,
    nrelation,
    ent_sample,
    center_mode,
    emb_dim,
    gamma,
):
    query_name_dict = {("e", ("r",)): "1p"}

    if model_name == "vec":
        model = GQE(
            nentity=nentity,
            nrelation=nrelation,
            #  emb_dim=800,
            #  gamma=24,
            emb_dim=emb_dim,
            gamma=gamma,
            model_name=model_name,
            test_batch_size=ent_sample,
            use_cuda=True,
            query_name_dict=query_name_dict,
        )
    elif model_name == "box":
        model = Query2Box(
            nentity=nentity,
            nrelation=nrelation,
            #  emb_dim=400,
            #  gamma=24,
            emb_dim=emb_dim,
            gamma=gamma,
            model_name=model_name,
            test_batch_size=ent_sample,
            center_mode=eval_tuple(center_mode),
            use_cuda=True,
            query_name_dict=query_name_dict,
        )
    elif model_name == "sphere":
        model = SpherE(
            nentity=nentity,
            nrelation=nrelation,
            #  emb_dim=256,
            #  gamma=24,
            emb_dim=emb_dim,
            gamma=gamma,
            model_name=model_name,
            center_mode=eval_tuple(center_mode),
            test_batch_size=ent_sample,
            use_cuda=True,
            query_name_dict=query_name_dict,
        )
    else:
        model = SpherE(
            nentity=nentity,
            nrelation=nrelation,
            #  emb_dim=256,
            #  gamma=24,
            emb_dim=emb_dim,
            gamma=gamma,
            model_name=model_name,
            center_mode=eval_tuple(center_mode),
            test_batch_size=ent_sample,
            use_cuda=True,
            query_name_dict=query_name_dict,
        )

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


class KGReasoning(nn.Module):
    def __init__(
        self, args, device, adj_list, query_name_dict, name_answer_dict, dataset_name
    ):
        super(KGReasoning, self).__init__()
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.device = device
        self.relation_embeddings = list()
        self.query_name_dict = query_name_dict
        self.name_answer_dict = name_answer_dict
        self.dataset_name = dataset_name

        self.fraction = args.fraction
        self.ent_sample = args.ent_sample

        self.thrshd = args.thrshd
        self.neg_scale = args.neg_scale

        self.model_name = args.model_name
        self.center_mode = args.center_mode
        self.emb_dim = args.hidden_dim
        self.gamma = args.gamma

        self.bucket = self.nentity // self.fraction
        self.rest = self.nentity - self.fraction * self.bucket
        self.neural_adj_path = args.neural_adj_path
        self.adj_list = adj_list
        self.pre_model = args.pre_model
        self.clqa_path = args.clqa_path
        self.kbc_path = args.kbc_path

        self.filename = ""
        self.cur_time = ""
        self.single_rel = args.single_rel
        self.logic_mode_p = args.logic_mode_p
        self.logic_mode = args.logic_mode

    def generate_neural_adj_matrix(self):
        """
        Load or generate neural adjacent matrix for all relations
        """
        logging.info(f"")
        logging.info(f"NEURAL ADJACENT MATRIX:")
        logging.info(f"Store matrix in path \t:{self.neural_adj_path}")
        logging.info(f"Entity sample size \t:{self.ent_sample}")
        logging.info(f"Fraction \t\t:{self.fraction}")
        logging.info(f"Threshold (epsilon) \t:{self.thrshd}\n")

        logging.info(f"AFTER NEURAL ADJACENT MATRIX:")
        logging.info(f"Logic mode p\t\t:{self.logic_mode_p}")
        logging.info(f"Logic mode \t\t:{self.logic_mode}")
        logging.info(f"Negation scaling \t:{self.neg_scale}")

        if not os.path.exists(self.neural_adj_path):
            try:
                os.makedirs(self.neural_adj_path)
            except Exception as e:
                logging.info(f"An error occurred while creating dir: {e}")
        self.cur_time = parse_time()
        self.filename = f"{self.dataset_name}_f{self.fraction}_b{self.ent_sample}_e{self.thrshd}_{self.pre_model}_{self.model_name}_{self.emb_dim}.pt"
        filename_read = os.path.join(self.neural_adj_path, self.filename)
        filename_write = os.path.join(
            self.neural_adj_path, self.cur_time, self.filename
        )
        logging.info(f"")
        if os.path.exists(filename_read):
            if self.single_rel:
                self.relation_embeddings = list()
            else:
                # This may take 30s - 2 mins (FB15k-237, sphere256)
                self.relation_embeddings = torch.load(
                    filename_read, map_location=self.device
                )
            logging.info(f"Loaded Relation Matrix '{filename_read}'\n")
        else:
            # load pre-trained models
            logging.info(f"Use pre model \t\t:{self.model_name}")
            logging.info(f"Embeddding dim \t:{self.emb_dim}")
            logging.info(f"In distance (optional) :{self.center_mode}")
            logging.info(f"Gamma \t\t\t:{self.gamma}")
            if self.pre_model == "clqa":
                model = load_clqa(
                    self.model_name,
                    self.clqa_path,
                    self.device,
                    self.nentity,
                    self.nrelation,
                    self.ent_sample,
                    self.center_mode,
                    self.emb_dim,
                    self.gamma,
                )
                logging.info(f"Loaded CLQA model '{self.clqa_path}' done!\n")
            logging.info(f"Generating Relation Matrix to {filename_write}...")

            # for clqa_forward
            # get batch_idxs_dict of queries
            batch_idxs_dict = collections.defaultdict(list)
            batch_idxs_dict_last = collections.defaultdict(list)
            ent_rest = self.nentity % self.ent_sample
            [batch_idxs_dict[("e", ("r",))].append(i) for i in range(self.ent_sample)]
            [batch_idxs_dict_last[("e", ("r",))].append(i) for i in range(ent_rest)]
            # all entity idxs
            negative_sample = (
                torch.arange(self.nentity)
                .unsqueeze(0)
                .expand(self.ent_sample, -1)
                .to(self.device)
            )

            if self.single_rel:
                self.neural_adj_path = os.path.join(self.neural_adj_path, self.cur_time)
                os.makedirs(self.neural_adj_path)
            for rel_idx in tqdm(range(self.nrelation)):
                relation_embedding = neural_adj_matrix(
                    model,
                    self.pre_model,
                    rel_idx,
                    self.nentity,
                    self.device,
                    self.thrshd,
                    self.adj_list[rel_idx],
                    self.ent_sample,
                    batch_idxs_dict,
                    batch_idxs_dict_last,
                    negative_sample,
                )
                # add fraction for efficiently store matrix on GPU memory (out-of-memory issue)
                fractional_relation_embedding = []
                for i in range(self.fraction):
                    step = i * self.bucket
                    step_next = (i + 1) * self.bucket
                    if i == self.fraction - 1:
                        step_next += self.rest
                    fractional_relation_embedding.append(
                        relation_embedding[step:step_next, :]
                        .to_sparse()
                        .to(self.device)
                    )
                # use singel_rel for out-of-memory issue
                if self.single_rel:
                    torch.save(
                        fractional_relation_embedding,
                        os.path.join(
                            self.neural_adj_path, f"{self.filename[:-3]}_{rel_idx}.pt"
                        ),
                    )
                else:
                    self.relation_embeddings.append(fractional_relation_embedding)
                # relation_embeddings is a list of a list of neural adjacent matrix,
                # shape of each matrix: (bucket, nentity)

            if not self.single_rel:
                os.makedirs(os.path.join(self.neural_adj_path, self.cur_time))
            torch.save(self.relation_embeddings, filename_write)
            logging.info(f"Generated Relation Matrix completed!\n")

        return self.relation_embeddings

    def relation_projection(self, embedding, r_embedding, is_neg=False):
        """
        Input:  embedding shape `(1, nentity)`
                r_embedding a list of a number `(fraction)` of tensor shape `(bucket, nentity)`
        Output: new embedding shape `(1, nentity)`
        """
        new_embedding = torch.zeros_like(embedding).to(self.device)  # (1, nentity)
        r_argmax = torch.zeros(self.nentity).to(self.device)
        for i in range(self.fraction):
            step = i * self.bucket
            step_next = (i + 1) * self.bucket
            if i == self.fraction - 1:
                step_next += self.rest

            # initialize each entity embedding
            fraction_embedding = embedding[:, step:step_next]  # (1, bucket)
            if fraction_embedding.sum().item() == 0:
                continue
            # get integer index of entity in between [0, bucket) with non-zero values
            nonzero = torch.nonzero(fraction_embedding, as_tuple=True)[1]  # (>=1)
            fraction_embedding = fraction_embedding[:, nonzero]  # (1, >=1)

            # select a fraction of neural adjacent matrix of a relation, for each entity in  # (bucket, nentity)
            # convert sparse tensor to dense tensor for performing operations
            # (1, >=1, nentity)
            fraction_r_embedding = r_embedding[i].to_dense()[nonzero, :].unsqueeze(0)
            if is_neg:
                # calculate minimum between (1, neg_scale * score)
                fraction_r_embedding = torch.minimum(
                    torch.ones_like(fraction_r_embedding).to(torch.float),
                    self.neg_scale * fraction_r_embedding,
                )
                fraction_r_embedding = 1.0 - fraction_r_embedding

            # relation projection
            if self.logic_mode_p == "prod":
                fraction_embedding_premax = (
                    fraction_r_embedding * fraction_embedding.unsqueeze(-1)
                )
            elif self.logic_mode_p == "godel":
                fraction_embedding_premax = torch.minimum(
                    fraction_r_embedding, fraction_embedding.unsqueeze(-1)
                )
            elif self.logic_mode_p == "lukas":
                fraction_embedding_premax = torch.maximum(
                    fraction_r_embedding + fraction_embedding.unsqueeze(-1) - 1.0,
                    torch.zeros_like(fraction_r_embedding),
                )
            else:
                fraction_embedding_premax = (
                    fraction_r_embedding * fraction_embedding.unsqueeze(-1)
                )

            #  fraction_embedding_premax                         # (1, >=1, nentity)
            fraction_embedding, tmp_argmax = torch.max(fraction_embedding_premax, dim=1)
            #  fraction_embedding, tmp_argmax                    # (1, nentity)

            tmp_argmax = nonzero[tmp_argmax.squeeze()] + step  # (nentity)
            mask_argmax = (
                (fraction_embedding > new_embedding).to(torch.long).squeeze()
            )  # (nentity)

            r_argmax = mask_argmax * tmp_argmax + (1 - mask_argmax) * r_argmax
            # r_argmax                                           # (nentity)

            # max value for new embedding of atomic query
            new_embedding = torch.maximum(new_embedding, fraction_embedding)

        return new_embedding, r_argmax.cpu().numpy()

    def intersection(self, embeddings):
        if self.logic_mode == "prod":
            return torch.prod(embeddings, dim=0)
        elif self.logic_mode == "godel":
            return torch.min(embeddings, dim=0)[0]
        elif self.logic_mode == "lukas":
            return torch.maximum(
                torch.sum(embeddings, dim=0) - 1.0,
                torch.zeros_like(torch.sum(embeddings, dim=0)).to(torch.float),
            )
        else:
            return torch.prod(embeddings, dim=0)

    def union(self, embeddings):
        if self.logic_mode == "prod":
            return 1.0 - torch.prod(1.0 - embeddings, dim=0)
        elif self.logic_mode == "godel":
            return torch.max(embeddings, dim=0)[0]
        elif self.logic_mode == "lukas":
            return torch.minimum(
                torch.sum(embeddings, dim=0),
                torch.ones_like(torch.sum(embeddings, dim=0)).to(torch.float),
            )
        else:
            return 1.0 - torch.prod(1.0 - embeddings, dim=0)

    def embed_query(self, queries, query_structure, idx):
        """
        Iterative embed a batch of queries with same structure
        queries: a flattened batch of queries
        """
        all_relation_flag = True
        exec_query = []
        # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
        for ele in query_structure[-1]:
            if ele not in ["r", "n"]:
                all_relation_flag = False
                break
        if all_relation_flag:
            # select an entity
            bsz = queries.size(0)
            if query_structure[0] == "e":
                embedding = (
                    torch.zeros(bsz, self.nentity).to(torch.float).to(self.device)
                )
                # assign 1 to entity element of embedding tensor at index queries[:, idx].unsqueeze(-1)
                embedding.scatter_(-1, queries[:, idx].unsqueeze(-1), 1)
                # append a constant entity
                exec_query.append(queries[:, idx].item())
                idx += 1
            else:
                embedding, idx, pre_exec_query = self.embed_query(
                    queries, query_structure[0], idx
                )
                exec_query.append(pre_exec_query)

            # relation projection
            r_exec_query = []
            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == "n":
                    assert (queries[:, idx] == -2).all()
                    r_exec_query.append("n")
                else:
                    # negation
                    # select a relation having an index "queries[0, idx]"
                    if self.single_rel:
                        local_path = os.path.join(
                            self.neural_adj_path,
                            f"{self.filename[:-3]}_{queries[0, idx]}.pt",
                        )
                        r_embedding = torch.load(local_path, map_location=self.device)
                        logging.info(f"Loaded matrix in '{local_path}' done!")
                    else:
                        r_embedding = self.relation_embeddings[queries[0, idx]]
                    if (i < len(query_structure[-1]) - 1) and query_structure[-1][
                        i + 1
                    ] == "n":
                        embedding, r_argmax = self.relation_projection(
                            embedding, r_embedding, True
                        )
                    # projection
                    else:
                        embedding, r_argmax = self.relation_projection(
                            embedding, r_embedding, False
                        )
                    r_exec_query.append((queries[0, idx].item(), r_argmax))
                    r_exec_query.append("e")
                idx += 1
            r_exec_query.pop()
            exec_query.append(r_exec_query)
            exec_query.append("e")
        else:
            embedding_list = []
            # check union query
            union_flag = False
            for ele in query_structure[-1]:
                if ele == "u":
                    union_flag = True
                    query_structure = query_structure[:-1]
                    break

            # start to embed a list of conjunctive queries
            for i in range(len(query_structure)):
                embedding, idx, pre_exec_query = self.embed_query(
                    queries, query_structure[i], idx
                )
                embedding_list.append(embedding)
                exec_query.append(pre_exec_query)

            # disjunction
            if union_flag:
                embedding = self.union(torch.stack(embedding_list))
                idx += 1
                exec_query.append(["u"])
            # conjunction
            else:
                embedding = self.intersection(torch.stack(embedding_list))
            exec_query.append("e")

        return embedding, idx, exec_query

    def find_ans(self, exec_query, query_structure, anchor):
        ans_structure = self.name_answer_dict[self.query_name_dict[query_structure]]

        return self.backward_ans(ans_structure, exec_query, anchor)

    def backward_ans(self, ans_structure, exec_query, anchor):
        #  (i+2)_th
        if ans_structure == "e":  # 'e'
            return exec_query, exec_query

        #  (i+2)_th
        elif ans_structure[0] == "u":  # 'u'
            return ["u"], "u"

        #  (i+2)_th
        elif ans_structure[0] == "r":  # ['r', 'e', 'r']
            cur_ent = anchor
            ans = []
            for ele, query_ele in zip(ans_structure[::-1], exec_query[::-1]):
                if ele == "r":
                    r_id, r_argmax = query_ele
                    ans.append(r_id)
                    cur_ent = int(r_argmax[cur_ent])
                elif ele == "n":
                    ans.append("n")
                else:
                    ans.append(cur_ent)
            return ans[::-1], cur_ent

        #  (i)_th
        elif ans_structure[1][0] == "r":  # [[...], ['r', ...], 'e']
            r_ans, r_ent = self.backward_ans(ans_structure[1], exec_query[1], anchor)
            e_ans, e_ent = self.backward_ans(ans_structure[0], exec_query[0], r_ent)
            ans = [e_ans, r_ans, anchor]
            return ans, e_ent

        #  (i+1)_th
        else:  # [[...], [...], 'e']
            ans = []
            for ele, query_ele in zip(ans_structure[:-1], exec_query[:-1]):
                ele_ans, ele_ent = self.backward_ans(ele, query_ele, anchor)
                ans.append(ele_ans)
            ans.append(anchor)
            return ans, ele_ent

from __future__ import division
from __future__ import print_function

import os
import time
import json
import argparse
import numpy as np

import torch
import torch.optim as optim
from collections import defaultdict
from torch.distributions import Beta

from utils import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='sgc', help='Graph encoder')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1600,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=9e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--num_tasks', type=int, default=5,
                    help='Number of meta-training tasks.')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Value of Beta distribution')
parser.add_argument('--intra', type=int, default=1,
                    help='Generate multiples')
parser.add_argument('--inter', type=int, default=5,
                    help='Generate tasks')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

print("this dataset is ", args.dataset)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

# Load data
dataset = args.dataset
adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(
    dataset) if args.dataset in ('Amazon_clothing', 'Amazon_eletronics', 'dblp') else load_cora_data()

# Model and optimizer
if args.encoder == 'gcn':
    encoder = GCN_Encoder(nfeat=features.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout)
    scorer = GCN_Valuator(nfeat=features.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout)
else:
    encoder = SGC_Encoder(nfeat=features.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout)

    scorer = SGC_Valuator(nfeat=features.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout)

optimizer_encoder = optim.Adam(encoder.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

optimizer_scorer = optim.Adam(scorer.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    encoder.cuda()
    scorer.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()


def train(class_selected, id_support, id_query, n_way, k_shot, q_qry, n_tasks):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    dist = Beta(torch.FloatTensor([0.5]), torch.FloatTensor([0.5]))

    loss_train = 0
    output_all, labels_all = [], []
    task_dict = defaultdict(dict)

    for i in range(n_tasks):
        # embedding lookup
        ori_support_embeddings = embeddings[id_support[i]]
        ori_support_embeddings = ori_support_embeddings.view([n_way, k_shot, z_dim])
        ori_query_embeddings = embeddings[id_query[i]]
        ori_query_embeddings = ori_query_embeddings.view([n_way, q_qry, z_dim])

        # node importance
        support_degrees = torch.log(degrees[id_support[i]].view([n_way, k_shot]))
        support_scores = scores[id_support[i]].view([n_way, k_shot])

        support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
        support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
        ori_support_embeddings = ori_support_embeddings * support_scores

        shuffle_id_s = np.arange(k_shot)
        np.random.shuffle(shuffle_id_s)
        shuffle_id_q = np.arange(q_qry)
        np.random.shuffle(shuffle_id_q)

        x2s = ori_support_embeddings[:, shuffle_id_s, :]
        x2q = ori_query_embeddings[:, shuffle_id_q, :]

        x_mix_s_list, x_mix_q_list = [], []
        for _ in range(args.intra):
            lam_mix = dist.sample().to(args.device)  # every lam_mix is different
            x_mix_s, _ = mixup_data(ori_support_embeddings, x2s, lam_mix)  # [n_way, k_shot, z_dim]
            x_mix_s_list.append(x_mix_s)
            x_mix_q, _ = mixup_data(ori_query_embeddings, x2q, lam_mix)  # [n_way, q_qry, z_dim]
            x_mix_q_list.append(x_mix_q)
        x_mix_s = torch.cat(x_mix_s_list, dim=1)
        x_mix_q = torch.cat(x_mix_q_list, dim=1)

        support_embeddings = torch.cat([ori_support_embeddings, x_mix_s], dim=1)
        query_embeddings = torch.cat([ori_query_embeddings, x_mix_q], dim=1)

        query_embeddings = query_embeddings.view(-1, z_dim)

        support_embeddings = support_embeddings.view(-1, z_dim)  # extra add, for cross-task mixup

        labels_new = torch.LongTensor([class_selected[i].index(j) for j in labels[id_query[i]]]).repeat_interleave(
            args.intra + 1)
        if args.cuda:
            labels_new = labels_new.cuda()

        task_dict[i] = {'spt': support_embeddings, 'qry': query_embeddings, 'lab': labels_new}

    for i in range(n_tasks):
        first_task = task_dict[i]
        second_id = (i + 1) % n_tasks
        second_task = task_dict[second_id]

        base = n_tasks + args.inter * i
        for j in range(args.inter):
            lam_inter = dist.sample().to(args.device)

            gen_task = cross_task(first_task, second_task, lam_inter, n_way, k_shot, q_qry)
            task_dict[base + j] = gen_task


    fin_task = len(task_dict)

    for k in range(fin_task):
        prototype_embeddings = task_dict[k]['spt'].view(n_way, -1, z_dim).sum(1)
        dists = euclidean_dist(task_dict[k]['qry'], prototype_embeddings)
        output = F.log_softmax(-dists, dim=1)

        loss_train += F.nll_loss(output, task_dict[k]['lab'])
        output_all.append(output)
        labels_all.append(task_dict[k]['lab'])

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    output_all = torch.cat(output_all)
    labels_all = torch.cat(labels_all)

    if args.cuda:
        output = output_all.cpu().detach()
        labels_new = labels_all.cpu().detach()

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def cross_task(task1, task2, lam_mix, n_way, k_shot, q_qry):
    new_task = dict()  # defaultdict(dict)
    task_2_shuffle_id = np.arange(n_way)
    update, update_eval = k_shot * (args.intra + 1), q_qry * (args.intra + 1)
    np.random.shuffle(task_2_shuffle_id)
    task_2_shuffle_id_s = np.array(
        [np.arange(update) + task_2_shuffle_id[idx] * update for idx in
         range(n_way)]).flatten()
    task_2_shuffle_id_q = np.array(
        [np.arange(update_eval) + task_2_shuffle_id[idx] * update_eval for
         idx in range(n_way)]).flatten()

    x2s = task2['spt'][task_2_shuffle_id_s]
    x2q = task2['qry'][task_2_shuffle_id_q]

    x_mix_s, _ = mixup_data(task1['spt'], x2s, lam_mix)

    x_mix_q, _ = mixup_data(task1['qry'], x2q, lam_mix)

    new_task['spt'] = x_mix_s
    new_task['qry'] = x_mix_q
    new_task['lab'] = task1['lab']  # labels

    return new_task


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])

    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':

    n_way = args.way
    k_shot = args.shot
    n_query = args.qry
    num_tasks = args.num_tasks
    meta_test_num = 50
    meta_valid_num = 50
    parameter = defaultdict(list)

    # Sampling a pool of tasks for validation/testing
    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query, 1) for i in
                  range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query, 1) for i in range(meta_test_num)]
    train_support, train_query, train_class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot,
                                                                      n_query, num_tasks)
    # Train model
    t_total = time.time()
    meta_train_acc = []

    best_test_acc = 0
    best_test_f1 = 0

    for episode in range(args.episodes):
        acc_train, f1_train = train(train_class_selected, train_support, train_query, n_way, k_shot, n_query, num_tasks)
        meta_train_acc.append(acc_train)
        if episode > 0 and episode % 50 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

            # # validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                      np.array(meta_test_f1).mean(axis=0)))
            # testing
            meta_test_acc = []
            meta_test_f1 = []

            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            fin_acc, fin_f1 = np.array(meta_test_acc).mean(axis=0), np.array(meta_test_f1).mean(axis=0)
            if fin_acc > best_test_acc:
                best_test_acc = fin_acc
                best_test_f1 = fin_f1
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(fin_acc, fin_f1))

    parameter[str((best_test_acc, best_test_f1))].append({'lr': args.lr, 'wd': args.weight_decay, \
                                                          'hidden': args.hidden, 'dropout': args.dropout,
                                                          'num_tasks': args.num_tasks})
    with open('{}_{}way_{}shot.json'.format(args.dataset, str(args.way), str(args.shot)), 'a', newline='\n') as f:
        json.dump(parameter, f)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Best-Test_Accuracy: {}, Meta-Test_F1: {}".format(best_test_acc, best_test_f1))

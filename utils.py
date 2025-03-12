import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
from dgl.data import CoraFullDataset

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(dataset_source):
    n1s = []
    n2s = []
    for line in open("../Meta-GPS++/few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s), max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_nodes, num_nodes))

    data_train = sio.loadmat("../Meta-GPS++/few_shot_data/{}_train.mat".format(dataset_source))
    train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

    data_test = sio.loadmat("../Meta-GPS++/few_shot_data/{}_test.mat".format(dataset_source))
    class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

    labels = np.zeros((num_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(features)

    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def load_cora_data():
    print("this is CoraFull")
    data = CoraFullDataset(raw_dir="../Meta-GPS++/few_shot_data/corafull")
    minus_node = [1, 4, 43, 68, 69]  # node number less than 70
    g = data[0]
    # features = torch.FloatTensor(normalize(g.ndata['feat'].numpy())).to(device) # test case
    features = g.ndata['feat'].to(device)
    label = g.ndata['label']
    np_label = label.numpy()
    label = label.to(device)

    degree = g.in_degrees()
    degree = torch.FloatTensor(degree.numpy())
    adj = g.adjacency_matrix(scipy_fmt='coo')
    adj_noloop = normalize_adj(adj)  # useless
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    class_list = []
    for cla in np_label:
        if cla not in class_list:
            class_list.append(cla)

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(np_label):
        id_by_class[cla].append(id)

    class_train = random.sample(class_list, 55)
    class_test = list(set(class_list).difference(set(class_train)))
    class_valid = random.sample(class_train, 15)
    class_train = list(set(class_train).difference(set(class_valid)))

    # minus less number node
    class_train = list(set(class_train).difference(set(minus_node)))
    class_valid = list(set(class_valid).difference(set(minus_node)))
    class_test = list(set(class_test).difference(set(minus_node)))

    return adj, features, label, degree, class_train, class_valid, class_test, id_by_class


def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='macro')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def task_generator(id_by_class, class_list, n_way, k_shot, m_query, n_tasks):
    if n_tasks == 1:
        # sample class indices
        class_selected = random.sample(class_list, n_way)
        id_support = []
        id_query = []
        for cla in class_selected:
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])
        return np.array(id_support), np.array(id_query), class_selected
    else:
        class_selected_fin = []
        id_support_fin = []
        id_query_fin = []
        for i in range(n_tasks):
            class_selected = random.sample(class_list, n_way)
            id_support = []
            id_query = []
            for cla in class_selected:
                temp = random.sample(id_by_class[cla], k_shot + m_query)
                id_support.extend(temp[:k_shot])
                id_query.extend(temp[k_shot:])
            class_selected_fin.append(class_selected)
            id_support_fin.append(id_support)
            id_query_fin.append(id_query)
        return np.array(id_support_fin), np.array(id_query_fin), class_selected_fin


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def mixup_data(xs, xq, lam):
    mixed_x = lam * xq + (1 - lam) * xs

    return mixed_x, lam

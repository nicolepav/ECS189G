'''
Concrete IO class for a specific dataset

@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.dataset import dataset

import torch
import numpy as np
import scipy.sparse as sp


class Dataset_Loader(dataset):

    def __init__(self, dName=None, dDescription=None, dPath=None):
        super(Dataset_Loader, self).__init__(dName, dDescription, dPath)

    def adj_normalize(self, mat):
        """normalize sparse matrix"""
        row_sum = np.array(mat.sum(1))
        r_inv = np.power(row_sum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mat = r_mat_inv.dot(mat).dot(r_mat_inv)
        return mat

    def sparse_mat_to_torch_sparse_tensor(self, sparse_mat):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mat = sparse_mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mat.data)
        shape = torch.Size(sparse_mat.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def generate_balanced_samples(self, idx, labels, idx_map, n_classes, n_train, n_test, n_val):
        shuffled_idx = idx.copy()
        rng = np.random.default_rng()
        rng.shuffle(shuffled_idx)
        n_nodes = round((n_train + n_test + n_val) / n_classes)

        # stands for current numbers and collections of [class_1, class_2, ..., class_n]
        class_counts = np.zeros(n_classes, dtype=np.int32)
        class_collections = np.zeros((n_classes, n_nodes), dtype=np.int32)

        for i in shuffled_idx:
            label = labels[np.where(idx == i)][0]

            if class_counts[label] < n_nodes:
                class_collections[label, class_counts[label]] = idx_map[i]
                class_counts[label] += 1

        idx_train = class_collections[:, :round(n_train / n_classes)].flatten()
        idx_test = class_collections[:, round(n_train / n_classes):round(n_train / n_classes) + round(n_test / n_classes)].flatten()
        idx_val = class_collections[:, round(n_train / n_classes) + round(n_test / n_classes):].flatten()
        rng.shuffle(idx_train)
        rng.shuffle(idx_test)
        rng.shuffle(idx_val)

        return idx_train, idx_test, idx_val

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mat_to_torch_sparse_tensor(norm_adj)

        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            idx_train, idx_test, idx_val = self.generate_balanced_samples(idx, labels, idx_map, 7, 140, 1400, 200)
        elif self.dataset_name == 'citeseer':
            idx_train, idx_test, idx_val = self.generate_balanced_samples(idx, labels, idx_map, 6, 120, 1200, 200)
        elif self.dataset_name == 'pubmed':
            idx_train, idx_test, idx_val = self.generate_balanced_samples(idx, labels, idx_map, 3, 60, 600, 200)
        else:
            print("The specified dataset does not have a predetermined training testing set partition")
            print("Please refer to src/dataset_loader.py to manually change it")
            print("Example: idx_train, idx_test, idx_val = self.generate_balanced_samples(idx, labels, 7, 140, 1400, 280)")
            print("Stands for dividing a 7 classes dataset into 140 nodes training set, 1400 nodes testing set, and 280 validating set")
            print("All subsets are balanced in classes")
            exit(1)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}

        print('{} dataset has been loaded!'.format(self.dataset_name))
        self.data = {'graph': graph, 'train_test_val': train_test_val}


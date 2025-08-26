import os
import ot
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def adata_hvg(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def adata_hvg_process(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.scale(adata)
    return adata

def adata_hvg_slide(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    return adata


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn
    #种子为2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_data(dataset, file_fold):
    if dataset == "DLPFC":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        # print("adata", adata)
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    else:
        platform = '10X'
        file_fold = os.path.join('../Data', platform, dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        df_meta = pd.read_csv(os.path.join('../Data', dataset,  'metadata.tsv'), sep='\t', header=None, index_col=0)
        adata.obs['layer_guess'] = df_meta['layer_guess']
        df_meta.columns = ['over', 'ground_truth']
        adata.obs['ground_truth'] = df_meta.iloc[:, 1]
        adata.var_names_make_unique()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    return adata, adata_X


def label_process_DLPFC(adata, df_meta):
    labels = df_meta["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    return adata


def graph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC':
        n = 12
        adj, edge_index = load_adj(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)
    else:
        n = 10
        adj, edge_index = load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    # return adata, adj, edge_index, adj2
    return adata, adj, edge_index, adj_mask


def load_adj(adata, n):
    adj = generate_adj(adata, include_self=False, n=n)
    # print("adj", adj)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # edge_index = adj_to_edge_index(adj)
    adj_norm, edge_index = preprocess_adj(adj)#
    return adj_norm, edge_index


def adj_to_edge_index(adj):
    dense_adj = adj.toarray()
    edge_index = torch.nonzero(torch.tensor(dense_adj), as_tuple=False).t()
    return edge_index


def load_adj2(adata, n):
    adj = generate_adj2(adata, include_self=True)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj_norm', adj)
    return adj


def load_adj_mask(adata, edge_index):
    N, E = adata.n_obs, edge_index.shape[1]
    adj_sm = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(N, N))
    adj_sm.fill_value_(1.)
    batch = torch.LongTensor(list(range(N)))
    batch, adj_batch = get_sim(batch, adj_sm, wt=50, wl=3)
    adj_mask = get_mask(adj_batch)
    return adj_mask


def generate_adj(adata, include_self=False, n=6):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj[i, n_neighbors] = 1
    if not include_self:
        x, y = np.diag_indices_from(adj)
        adj[x, y] = 0
    adj = adj + adj.T
    adj = adj > 0
    adj = adj.astype(np.int64)
    return adj


def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    # edge_index = adj_to_edge_index(adj)
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    edge_index = adj_to_edge_index(adj_normalized)

    return sparse_mx_to_torch_sparse_tensor(adj_normalized), edge_index


def generate_adj2(adata, include_self=True):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    dist = dist / np.max(dist)
    adj = dist.copy()
    if not include_self:
        np.fill_diagonal(adj, 0)
    # print('adj', adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_sim(batch, adj, wt=100, wl=2):
    rowptr, col, _ = adj.csr()
    batch_size = batch.shape[0]
    batch_repeat = batch.repeat(wt)
    rw = adj.random_walk(batch_repeat, wl)[:, 1:]

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    rw = rw.t().reshape(-1, batch_size).t()
    row, col, val = [], [], []
    for i in range(batch.shape[0]):
        rw_nodes, rw_times = torch.unique(rw[i], return_counts=True)
        row += [batch[i].item()] * rw_nodes.shape[0]
        col += rw_nodes.tolist()
        val += rw_times.tolist()

    unique_nodes = list(set(row + col))
    subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

    row = [subg2g[x] for x in row]
    col = [subg2g[x] for x in col]
    idx = torch.tensor([subg2g[x] for x in batch.tolist()])

    adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                        sparse_sizes=(len(unique_nodes), len(unique_nodes)))

    adj_batch, _ = adj_.saint_subgraph(idx)
    adj_batch = adj_batch.set_diag(0.)
    return batch, adj_batch


def get_mask(adj):
    batch_mean = adj.mean(dim=1)
    mean = batch_mean[torch.LongTensor(adj.storage.row())]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_


def diffusion_adj(norm_adj, mode="ppr", transport_rate=0.2):
    device = norm_adj.device  
    n = norm_adj.shape[0]
    if mode == "ppr":
        diff_adj = transport_rate * torch.linalg.inv(torch.eye(n, device=device) - (1 - transport_rate) * norm_adj)

    return diff_adj


def edge_index_to_sparse_adj(edge_index, num_nodes, device='cuda'):
    edge_weight = torch.ones(edge_index.size(1), device=device)
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    )
    return adj


def to_sparse_tensor(adj_matrix):
    edge_index, edge_attr = dense_to_sparse(adj_matrix)
    edge_index = edge_index.long()
    num_nodes = adj_matrix.shape[0]

    return SparseTensor.from_edge_index(edge_index, edge_attr, sparse_sizes=(num_nodes, num_nodes))


def scale_diff(adj_diff):
    edge_index1 = adj_diff.nonzero(as_tuple=False).T  
    edge_weight = adj_diff[edge_index1[0], edge_index1[1]]
    num_nodes = adj_diff.shape[0]
    adj_diff = torch.sparse_coo_tensor(edge_index1, edge_weight, (num_nodes, num_nodes))
    adj_diff = adj_diff.to(adj_diff.device)

    threshold = 1e-3  
    mask = adj_diff._values() > threshold 

    filtered_indices = adj_diff._indices()[:, mask]
    filtered_values = adj_diff._values()[mask]

    adj_diff = torch.sparse_coo_tensor(filtered_indices, filtered_values, adj_diff.shape).coalesce()

    return adj_diff



def refine_label(adata, radius=50, key='label'):  
    n_neigh = radius  
    new_type = [] 
    old_type = adata.obs[key].values  

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')  

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def cross_correlation(Z_v1, Z_v2):

    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):

    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

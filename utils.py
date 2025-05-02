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

    elif dataset == "Human_Breast_Cancer":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == "Adult_Mouse_Brain_Section_1":
        adata = sc.read_visium(file_fold, count_file='V1_Adult_Mouse_Brain_Coronal_Section_1_filtered_feature_bc_matrix.h5', load_images=True)
        # print('adata', adata)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == "Mouse_Brain_Anterior_Section1":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        # print('adata', adata)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == "ME":
        adata = sc.read_h5ad(file_fold + 'E9.5_E1S1.MOSTA.h5ad')
        print('adata', adata)
        adata.var_names_make_unique()
        # # print("adata", adata)
        # adata.obs['x'] = adata.obs["array_row"]
        # adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == 'MOB':
        savepath = '../Result/MOB_Stereo/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        counts_file = os.path.join(file_fold, 'RNA_counts.tsv')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0).T
        counts.index = [f'Spot_{i}' for i in counts.index]
        adata = sc.AnnData(counts)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()

        pos_file = os.path.join(file_fold, 'position.tsv')
        coor_df = pd.read_csv(pos_file, sep='\t')
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]
        # print('adata.obs_names', adata.obs_names)
        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obs['x'] = coor_df['x'].tolist()
        adata.obs['y'] = coor_df['y'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        print(adata)

        # hires_image = os.path.join(file_fold, 'crop1.png')
        # adata.uns["spatial"] = {}
        # adata.uns["spatial"][dataset] = {}
        # adata.uns["spatial"][dataset]['images'] = {}
        # adata.uns["spatial"][dataset]['images']['hires'] = imread(hires_image)

        # label_file = pd.read_csv(os.path.join(file_fold, 'Cell_GetExp_gene.txt'), sep='\t', header=None)
        # used_barcode = label_file[0]

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()

        # adata.obs['total_exp'] = adata.X.sum(axis=1)
        # fig, ax = plt.subplots()
        # sc.pl.spatial(adata, color='total_exp', spot_size=40, show=False, ax=ax)
        # ax.invert_yaxis()
        # plt.savefig(savepath + 'STMCCL_stereo_MOB1.jpg', dpi=600)

        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        # print("adata是否降维", adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))

    elif dataset == 'MOB_V2':
        # savepath = '../Result/MOB_Slide/_0.005/'
        # if not os.path.exists(savepath):
        #     os.mkdir(savepath)
        # counts_file = os.path.join(input_dir, '')
        # coor_file = os.path.join(input_dir, '')
        counts_file = os.path.join(file_fold, 'Puck_200127_15.digital_expression.txt')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        adata = sc.AnnData(counts.T)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()
        print(adata)

        coor_file = os.path.join(file_fold, 'Puck_200127_15_bead_locations.csv')
        coor_df = pd.read_csv(coor_file, index_col=0)
        # coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.set_index('barcode')
        coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
        adata.obs['x'] = coor_df['xcoord'].tolist()
        adata.obs['y'] = coor_df['ycoord'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        # print("adata", adata)
        # plt.rcParams["figure.figsize"] = (6, 5)
        # # Original tissue area, some scattered spots
        # sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", s=6, show=False, save='_MOB01_slide.png')
        # plt.title('')
        # plt.axis('off')
        # plt.savefig(savepath + 'STNMAE_MOBV2.jpg', dpi=300)

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))

    elif dataset == 'hip':
        savepath = '../Result/hip/_0.005/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        # counts_file = os.path.join(input_dir, '')
        # coor_file = os.path.join(input_dir, '')
        counts_file = os.path.join(file_fold, 'Puck_200115_08.digital_expression.txt')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        adata = sc.AnnData(counts.T)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()
        print(adata)

        coor_file = os.path.join(file_fold, 'Puck_200115_08_bead_locations.csv')
        coor_df = pd.read_csv(coor_file, index_col=0)
        # coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.set_index('barcode')
        coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
        adata.obs['x'] = coor_df['xcoord'].tolist()
        adata.obs['y'] = coor_df['ycoord'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        print(adata)
        plt.rcParams["figure.figsize"] = (6, 5)
        sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", s=6, show=False)
        plt.title('prime')
        plt.axis('off')
        plt.savefig(savepath + 'STMCCL_hip_1.jpg', dpi=600)

        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_slide(adata)
        # print("adata是否降维", adata)
        print("adata", adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))


    elif dataset == 'ISH':
        adata = sc.read(file_fold + '/STARmap_20180505_BY3_1k.h5ad')
        # print(adata)
        adata.obs['x'] = adata.obs["X"]
        adata.obs['y'] = adata.obs["Y"]
        adata.layers['count'] = adata.X
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == 'mouse_somatosensory_cortex':
        adata = sc.read(file_fold + '/osmFISH_cortex.h5ad')
        print(adata)
        adata.var_names_make_unique()
        adata = adata[adata.obs["Region"] != "Excluded"]

        adata.layers['count'] = adata.X
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        # print("adata1", adata)
        # adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata_X = adata.X
        adata.obsm['X_pca'] = adata.X

    else:
        platform = '10X'
        file_fold = os.path.join('../Data', platform, dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        df_meta = pd.read_csv(os.path.join('../Data', dataset,  'metadata.tsv'), sep='\t', header=None, index_col=0)
        adata.obs['layer_guess'] = df_meta['layer_guess']
        df_meta.columns = ['over', 'ground_truth']
        adata.obs['ground_truth'] = df_meta.iloc[:, 1]#同理获取获取数据

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


def label_process_HBC(adata, df_meta):
    labels = df_meta["ground_truth"].copy()
    # print("labels", labels)
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground = ground.replace('DCIS/LCIS_1', '0')
    ground = ground.replace('DCIS/LCIS_2', '1')
    ground = ground.replace('DCIS/LCIS_4', '2')
    ground = ground.replace('DCIS/LCIS_5', '3')
    ground = ground.replace('Healthy_1', '4')
    ground = ground.replace('Healthy_2', '5')
    ground = ground.replace('IDC_1', '6')
    ground = ground.replace('IDC_2', '7')
    ground = ground.replace('IDC_3', '8')
    ground = ground.replace('IDC_4', '9')
    ground = ground.replace('IDC_5', '10')
    ground = ground.replace('IDC_6', '11')
    ground = ground.replace('IDC_7', '12')
    ground = ground.replace('IDC_8', '13')
    ground = ground.replace('Tumor_edge_1', '14')
    ground = ground.replace('Tumor_edge_2', '15')
    ground = ground.replace('Tumor_edge_3', '16')
    ground = ground.replace('Tumor_edge_4', '17')
    ground = ground.replace('Tumor_edge_5', '18')
    ground = ground.replace('Tumor_edge_6', '19')
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground.values.astype(int)
    # print("ground", adata.obs['ground'])
    return adata


def label_process_Mouse_brain_anterior(adata, df_meta):
    labels = df_meta["ground_truth"].copy()
    # print("labels", labels)
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground = ground.replace('AOB::Gl', '0')
    ground = ground.replace('AOB::Gr', '1')
    ground = ground.replace('AOB::Ml', '2')
    ground = ground.replace('AOE', '3')
    ground = ground.replace('AON::L1_1', '4')
    ground = ground.replace('AON::L1_2', '5')
    ground = ground.replace('AON::L2', '6')
    ground = ground.replace('AcbC', '7')
    ground = ground.replace('AcbSh', '8')
    ground = ground.replace('CC', '9')
    ground = ground.replace('CPu', '10')
    ground = ground.replace('Cl', '11')
    ground = ground.replace('En', '12')
    ground = ground.replace('FRP::L1', '13')
    ground = ground.replace('FRP::L2/3', '14')
    ground = ground.replace('Fim', '15')
    ground = ground.replace('Ft', '16')
    ground = ground.replace('HY::LPO', '17')
    ground = ground.replace('Io', '18')
    ground = ground.replace('LV', '19')
    ground = ground.replace('MO::L1', '20')
    ground = ground.replace('MO::L2/3', '21')
    ground = ground.replace('MO::L5', '22')
    ground = ground.replace('MO::L6', '23')
    ground = ground.replace('MOB::Gl_1', '24')
    ground = ground.replace('MOB::Gl_2', '25')
    ground = ground.replace('MOB::Gr', '26')
    ground = ground.replace('MOB::MI', '27')
    ground = ground.replace('MOB::Opl', '28')
    ground = ground.replace('MOB::lpl', '29')
    ground = ground.replace('Not_annotated', '30')
    ground = ground.replace('ORB::L1', '31')
    ground = ground.replace('ORB::L2/3', '32')
    ground = ground.replace('ORB::L5', '33')
    ground = ground.replace('ORB::L6', '34')
    ground = ground.replace('OT::Ml', '35')
    ground = ground.replace('OT::Pl', '36')
    ground = ground.replace('OT::PoL', '37')
    ground = ground.replace('Or', '38')
    ground = ground.replace('PIR', '39')
    ground = ground.replace('Pal::GPi', '40')
    ground = ground.replace('Pal::MA', '41')
    ground = ground.replace('Pal::NDB', '42')
    ground = ground.replace('Pal::Sl', '43')
    ground = ground.replace('Py', '44')
    ground = ground.replace('SLu', '45')
    ground = ground.replace('SS::L1', '46')
    ground = ground.replace('SS::L2/3', '47')
    ground = ground.replace('SS::L5', '48')
    ground = ground.replace('SS::L6', '49')
    ground = ground.replace('St', '50')
    ground = ground.replace('TH::RT', '51')
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground.values.astype(int)
    # print("ground", adata.obs['ground'])
    return adata


def graph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC_gin':
        n = 12
        adj, edge_index = load_adj(adata, n)#加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        # # adj2 = load_adj2(adata, n)#返回没有加上自环且标准化的邻接矩阵（这个矩阵的边权重不是1而且某种计算权重）
        adj_mask = load_adj_mask(adata, edge_index)

    elif dataset == 'MBO':
        n = 10
        adj, edge_index = load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    elif dataset =='MOB_V2':
        n = 7
        adj, edge_index = load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    elif dataset == 'hip':
        n = 100
        adj, edge_index = load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    elif dataset == 'Adult_Mouse_Brain_Section_1':
        n = 5
        adj, edge_index = load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    elif dataset == 'ISH':
        n = 7
        adj, edge_index= load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    else:
        n = 10
        adj, edge_index = load_adj(adata, n)
        # adj2 = load_adj2(adata, n)
        adj_mask = load_adj_mask(adata, edge_index)

    # return adata, adj, edge_index, adj2
    return adata, adj, edge_index, adj_mask


def load_adj(adata, n):
    adj = generate_adj(adata, include_self=False, n=n)#生成邻接矩阵
    # print("adj", adj)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()#去除零值
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
    device = norm_adj.device  # 保持设备一致
    n = norm_adj.shape[0]
    # 计算扩散邻接矩阵
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
    edge_index1 = adj_diff.nonzero(as_tuple=False).T  # 转置，变成 (2, num_edges)
    edge_weight = adj_diff[edge_index1[0], edge_index1[1]]
    num_nodes = adj_diff.shape[0]
    adj_diff = torch.sparse_coo_tensor(edge_index1, edge_weight, (num_nodes, num_nodes))
    adj_diff = adj_diff.to(adj_diff.device)

    threshold = 1e-3  # 设定一个阈值
    mask = adj_diff._values() > threshold  # 仅保留大于 threshold 的值

    # 🔹 过滤出符合条件的索引和值
    filtered_indices = adj_diff._indices()[:, mask]
    filtered_values = adj_diff._values()[mask]

    # 🔹 重新创建更小的稀疏张量
    adj_diff = torch.sparse_coo_tensor(filtered_indices, filtered_values, adj_diff.shape).coalesce()

    return adj_diff



def refine_label(adata, radius=50, key='label'):  # 修正函数相当于强制修正，
    # 功能，使得每个spot半径小于50的范围内，其他spot 的大部分是哪一类就把这个spot 强制归为这一类。
    n_neigh = radius  # 定义半径
    new_type = []  # spot新的类型
    old_type = adata.obs[key].values  ##读入数据的原始类型

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')  # 用欧氏距离

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

    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())#计算不同视图相关性矩阵，S[i, j] 表示 Z_v1[i] 和 Z_v2[j] 之间的余弦相似度。


def correlation_reduction_loss(S):

    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# def clustering(feature, n_clusters, true_labels, kmeans_device='cpu', batch_size=100000, tol=1e-4, device=torch.device('cuda:0'), spectral_clustering=False):
#     if spectral_clustering:
#         if isinstance(feature, torch.Tensor):
#             feature = feature.numpy()
#         print("spectral clustering on cpu...")
#         patch_sklearn()
#         Cluster = SpectralClustering(
#             n_clusters=n_clusters, affinity='precomputed', random_state=0)
#         f_adj = np.matmul(feature, np.transpose(feature))
#         predict_labels = Cluster.fit_predict(f_adj)
#     else:
#         if kmeans_device == 'cuda':
#             if isinstance(feature, np.ndarray):
#                 feature = torch.tensor(feature)
#             print("kmeans on gpu...")
#             predict_labels, _ = kmeans(
#                 X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
#             predict_labels = predict_labels.numpy()
#         else:
#             if isinstance(feature, torch.Tensor):
#                 feature = feature.numpy()
#             print("kmeans on cpu...")
#             patch_sklearn()
#             Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
#             predict_labels = Cluster.fit_predict(feature)
#
#     cm = clustering_metrics(true_labels, predict_labels)
#     acc, nmi, adjscore, fms, f1_macro, f1_micro = cm.evaluationClusterModelFromLabel(tqdm)
#     return acc, nmi, adjscore, f1_macro, f1_micro


def clustering(y, Z, n_clusters):
    model = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z)
    acc, f1 = eva(y, cluster_id)
    return acc, f1


def eva(y_true, y_pred):
    acc, f1 = cluster_acc(y_true, y_pred)
    return acc, f1


def cluster_acc(y_true, y_pred):

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro
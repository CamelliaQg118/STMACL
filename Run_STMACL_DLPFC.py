import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
import scipy as sp
import numpy as np
import torch
import copy
import os
import STMACL

import utils

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


ARI_list = []
random_seed = 42
STMACL.fix_seed(random_seed)
os.environ['R_HOME'] = 'D:/R/R-4.3.3/R-4.3.3'
os.environ['R_USER'] = 'D:/Anaconda3/Anaconda3202303/envs/STMACL/Lib/site-packages/rpy2'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

dataset = 'DLPFC'
slice = '151676'
platform = '10X'
file_fold = os.path.join('../Data', platform, dataset, slice)
adata, adata_X = utils.load_data(dataset, file_fold)
print("adata", adata)

df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
adata = utils.label_process_DLPFC(adata, df_meta)

savepath = '../Result/refer/DLPFC/' + str(slice) + '/'


if not os.path.exists(savepath):
    os.mkdir(savepath)
n_clusters = 5 if slice in ['151669', '151670', '151671', '151672'] else 7

adata, adj, edge_index, adj_mask = utils.graph_build(adata, adata_X, dataset)
stmacl_net = STMACL.stmacl(adata.obsm['X_pca'], adata, adj, edge_index, adj_mask, n_clusters, dataset, device=device)

tool = None
if tool == 'mclust':
    emb = stmacl_net.train()
    adata.obsm['STMACL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STMACL.mclust_R(adata, n_clusters, use_rep='STMACL', key_added='STMACL', random_seed=random_seed)
elif tool == 'leiden':
    emb = stmacl_net.train()
    adata.obsm['STMACL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STMACL.leiden(adata, n_clusters, use_rep='STMACL', key_added='STMACL', random_seed=random_seed)
elif tool == 'louvain':
    emb = stmacl_net.train()
    adata.obsm['STMACL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STMACL.louvain(adata, n_clusters, use_rep='STMACL', key_added='STMACL', random_seed=random_seed)
else:
    # emb, idx = stmacl_net.train()
    emb, idx = stmacl_net.train()
    print("emb", emb)
    adata.obsm['STMACL'] = emb
    adata.obs['STMACL'] = idx
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

print("adata", adata)
new_type = utils.refine_label(adata, radius=10, key='STMACL')
adata.obs['STMACL'] = new_type
ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STMACL'])
NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STMACL'])
adata.uns["ARI"] = ARI
adata.uns["NMI"] = NMI

print('===== Project: {}_{} ARI score: {:.4f}'.format(str(dataset), str(slice), ARI))
print('===== Project: {}_{} NMI score: {:.4f}'.format(str(dataset), str(slice), NMI))
print(str(slice))
print(n_clusters)
ARI_list.append(ARI)

plt.rcParams["figure.figsize"] = (3, 3)
title = "Manual annotation (" + dataset + "#" + slice + ")"
sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=300)
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
sc.pl.spatial(adata, color='ground_truth', ax=axes[0], show=False)
sc.pl.spatial(adata, color=['STMACL'], ax=axes[1], show=False)
axes[0].set_title("Manual annotation ("+ slice + ")")
axes[1].set_title('ARI=%.4f, NMI=%.4f' % (ARI, NMI))

plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)  
plt.savefig(savepath + 'STMACL.jpg', dpi=300)  


sc.pp.neighbors(adata, use_rep='STMACL', metric='cosine')
sc.tl.umap(adata)
sc.pl.umap(adata, color='STMACL', title='STMACL', show=False)
plt.savefig(savepath + 'umap.jpg', bbox_inches='tight', dpi=300)

for ax in axes:
    ax.set_aspect(1)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)


title = 'STMACL:{}_{} ARI={:.4f} NMI={:.4f} '.format(str(dataset), str(slice), adata.uns['ARI'],
                                                                         adata.uns['NMI'])
sc.pl.spatial(adata, img_key="hires", color=['STMACL'], title=title, show=False)
plt.savefig(savepath + 'STMACL_NMI_ARI_acc_f1.tif', bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.tl.paga(adata, groups='STMACL')
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2, show=False)
plt.savefig(savepath + 'STMACL_PAGA_domain.tif', bbox_inches='tight', dpi=300)

sc.tl.paga(adata, groups='ground_truth')
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2, show=False)
plt.savefig(savepath + 'STMACL_PAGA_ground_truth.png', bbox_inches='tight', dpi=300)




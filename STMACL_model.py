import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .STMACL_module import stmacl_module
from tqdm import tqdm
from sklearn import metrics
import STMACL
from utils import scale
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
EPS = 1e-15


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def kl_loss(q, p):
    return F.kl_div(q, p, reduction="batchmean")


def cos(out, mask, temperature=0.2, scale_by_temperature=True):
    device = (torch.device('cuda') if out.is_cuda else torch.device('cpu'))
    row, col, val = mask.storage.row().to(device), mask.storage.col().to(device), mask.storage.value().to(device)
    batch_size = out.shape[0]

    # compute logits
    dot = torch.matmul(out, out.T)
    dot = torch.div(dot, temperature)

    # for numerical stability
    logits_max, _ = torch.max(dot, dim=1, keepdim=True)
    dot = dot - logits_max.detach()

    # for numerical stability
    logits_mask = torch.scatter(torch.ones(batch_size, batch_size).to(device), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
    exp_logits = torch.exp(dot) * logits_mask
    log_probs = dot - torch.log(exp_logits.sum(1, keepdim=True))

    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has NaN!")

    labels = row.view(row.shape[0], 1)
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

    log_probs = log_probs[row, col].view(-1, 1)
    loss = torch.zeros_like(unique_labels, dtype=torch.float).to(device)
    loss.scatter_add_(0, labels, log_probs)
    loss = -loss / labels_count.float().unsqueeze(1)

    if scale_by_temperature:
        loss *= temperature
    loss = loss.mean()

    return loss.mean()


class stmacl:
    def __init__(
            self,
            X,
            adata,
            adj,
            edge_index,
            adj_mask,
            n_clusters,
            dataset,
            rec_w=10,
            icr_w=1,
            adj_w=3,
            cos_w=1,
            kl_w=1,
            dec_tol=0.00,
            threshold=0.5,
            epochs=600,
            dec_interval=3,
            lr=0.0001,
            decay=0.0001,
            device='cuda:0',
            mode='clustering',
    ):
        self.random_seed = 42
        STMACL.fix_seed(self.random_seed)

        self.n_clusters = n_clusters
        self.rec_w = rec_w
        self.icr_w = icr_w
        self.adj_w = adj_w
        self.cos_w = cos_w
        self.kl_w = kl_w


        self.device = device
        self.dec_tol = dec_tol
        self.threshold = threshold 

        self.adata = adata.copy()
        self.dataset = dataset
        self.cell_num = len(X)
        self.epochs = epochs
        self.dec_interval = dec_interval
        self.learning_rate = lr
        self.weight_decay = decay
        self.adata = adata.copy()
        # self.X = torch.FloatTensor(X.copy()).to(self.device)
        # self.input_dim = self.X.shape[1]
        self.adj = adj.to(self.device)
        self.adj_mask = adj_mask.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.mode = mode

        if self.mode == 'clustering':
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        elif self.mode == 'imputation':
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise Exception
        self.input_dim = self.X.shape[-1]

        self.model = stmacl_module(self.adj, self.input_dim, self.n_clusters).to(self.device)

    def train(self, dec_tol=0.00):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_clusters * 2, random_state=42)
        emb, rec, q, loss_rec, loss_adj, loss_icr = self.model_eval()
        y_pred_last = np.copy(kmeans.fit_predict(emb))
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()
        list_rec = []
        list_de = []
        list_icr = []
        list_adj = []
        list_cos = []
        list_kl = []
        epoch_max = 0
        ari_max = 0
        idx_max = []
        emb_max = []

        if self.dataset in ['DLPFC']:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                if epoch % self.dec_interval == 0:
                    emb, rec, tmp_q, loss_rec, loss_adj, loss_icr = self.model_eval()
                    tmp_p = target_distribution(torch.Tensor(tmp_q))
                    y_pred = tmp_p.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)
                emb, rec, out_q, loss_rec, loss_adj, loss_icr = self.model(self.X, self.adj, self.edge_index)
                loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                Liner = torch.nn.Linear(emb.shape[1], emb.shape[1], bias=False).to(self.device)
                # out = scale(emb)
                out = Liner(emb)
                out = F.normalize(out, p=2, dim=1)
                loss_cos = cos(out, self.adj_mask, temperature=0.2)
                loss_tatal = self.rec_w * loss_rec + self.icr_w * loss_icr + \
                             self.adj_w * loss_adj + self.kl_w * loss_kl + self.cos_w * loss_cos
                # loss_tatal = self.rec_w * loss_rec + self.adj_w * loss_adj + self.cos_w * loss_cos

                loss_tatal.backward()
                self.optimizer.step()

                emb, _, _, _, _, _ = self.model_eval()
                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
                idx = kmeans.labels_
                self.adata.obsm['STMACL'] = emb
                # adata1 = ST_NMAE.mclust_R(self.adata, self.n_clusters, use_rep='STNMAE', key_added='STNMAE', random_seed=self.random_seed)
                labels = self.adata.obs['ground']
                labels = pd.to_numeric(labels, errors='coerce')
                labels = pd.Series(labels).fillna(0).to_numpy()
                idx = pd.Series(idx).fillna(0).to_numpy()

                ari_res = metrics.adjusted_rand_score(labels, idx)
                if ari_res > ari_max:
                    ari_max = ari_res
                    epoch_max = epoch
                    idx_max = idx
                    emb_max = emb
                    rec1_max = rec

            print("epoch_max", epoch_max)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx_max)
            self.adata.obs['STMACL'] = idx_max.astype(str)
            self.adata.obsm['emb'] = emb_max
            # self.adata.obsm['rec'] = rec1_max
            return self.adata.obsm['emb'], self.adata.obs['STMACL']
            # return emb
        else:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                if epoch % self.dec_interval == 0:
                    emb, rec, tmp_q, loss_rec, loss_adj, loss_icr = self.model_eval()
                    tmp_p = target_distribution(torch.Tensor(tmp_q))
                    y_pred = tmp_p.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)

                out_emb, rec, out_q, loss_rec, loss_adj, loss_icr = self.model(self.X, self.adj, self.edge_index)
                loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                Liner = torch.nn.Linear(out_emb.shape[1], out_emb.shape[1], bias=False).to(self.device)
                # out = scale(emb)
                out = Liner(out_emb)
                out = F.normalize(out, p=2, dim=1)
                loss_cos = cos(out, self.adj_mask, temperature=0.2)

                loss_tatal = self.rec_w * loss_rec + self.icr_w * loss_icr + \
                             self.adj_w * loss_adj + self.kl_w * loss_kl + self.cos_w * loss_cos

                loss_tatal.backward()
                self.optimizer.step()

            return emb# #rec

    def model_eval(self):
        self.model.eval()
        emb, rec, q, loss_rec, loss_adj, loss_icr = self.model(self.X, self.adj, self.edge_index)
        emb = emb.data.cpu().numpy()
        rec = rec.data.cpu().numpy()
        q = q.data.cpu().numpy()
        loss_rec = loss_rec.data.cpu().numpy()
        loss_icr = loss_icr.data.cpu().numpy()
        loss_adj = loss_adj.data.cpu().numpy()
        return emb, rec, q, loss_rec, loss_adj, loss_icr




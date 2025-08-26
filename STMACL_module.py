from functools import partial
import layers
import utils
from torch_geometric.utils import negative_sampling
import numpy as np
import torch.backends.cudnn as cudnn
from STMACL.layers import *
cudnn.deterministic = True
cudnn.benchmark = True


def sce_loss(x, y, alpha=3):

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = layers.GraphConvolution(input_dim, hidden_dim)
        self.gc2 = layers.GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = nn.PReLU()(self.gc1(x, adj))  
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class ZINB_decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINB_decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ELU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        print("X", x)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]
        if reduction:
            x = x.mean(1)
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


def zinb_loss(y_true, y_pred, theta, pi, scale_factor=1.0, ridge_lambda=0.0, eps=1e-10, mean=True):

    theta = torch.minimum(theta, torch.tensor(1e6))

    y_pred = y_pred * scale_factor
    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))
    nb_loss = t1 + t2

    if pi is None:  
        loss = nb_loss
    else:  
        nb_case = nb_loss - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        loss = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)

        ridge = ridge_lambda * torch.square(pi)
        loss += ridge

    if mean:
        loss = torch.mean(loss)

    return loss


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = np.log(2.)
    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss(l_enc, g_enc, measure):

    num_nodes = l_enc.shape[0]  
    pos_mask = torch.eye(num_nodes).cuda()  
    neg_mask = 1 - pos_mask  

    if g_enc.dim() == 1:
        g_enc = g_enc.unsqueeze(0)
    res = torch.mm(l_enc, g_enc.T)  
    
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes 
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_nodes - 1))  

    return E_neg - E_pos


def bcn_loss(out1, out2):
    loss = F.binary_cross_entropy(out1.sigmoid(), torch.ones_like(out2))
    return loss


class Attention(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(input_dim,  output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1, bias=False)
        )

    def forward(self, z):

        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
        

class stmacl_module(nn.Module):
    def __init__(
            self,
            adj,
            input_dim,
            nclass,
            latent_dim=128,
            output_dim=64,
            train_dim=128,
            num_layers=2,
            p_drop=0.2,
            edcode_type='GCN',
            g_type='GCN',
            decode_type='GCN',
            remask_method="random",
            dorp_code=0.2,
            drop_en=0.2,
            drop_edge_rate=0.1,
            alpha=0.1,
            beta=0.2,
            mask_rate=0.8,
            remask_rate=0.8,
            edmask_rate=0.7,
            device='cuda:0'
    ):
        super(stmacl_module, self).__init__()
        self.device = device
        self.adj_dim = adj.shape[0]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_dim = train_dim
        self.edin_dim = input_dim
        self.edlatent_dim = latent_dim
        self.edout_dim = output_dim
        self.en_hidden = latent_dim
        self.zinb_indim = input_dim
        self.zinb_latentdim = latent_dim
        self.zinb_outdim = output_dim
        self.ed_indim = output_dim
        self.ed_latentdim = latent_dim
        self.ed_outdim = output_dim
        self.pm_indim = input_dim
        self.pm_latentdim = latent_dim
        self.pm_outdim = output_dim
        self.el_indim = output_dim
        self.el_latentdim = latent_dim
        self.el_outdim = output_dim
        self.mask_dim = output_dim
        self.dein_dim = output_dim
        self.deout_dim = output_dim
        self.eg_indim = input_dim
        self.eg_latentdim = latent_dim
        self.eg_outdim = output_dim
        self.p_latent = output_dim
        self.emb_dim = output_dim*3
        self.g_type = g_type
        self.eg_type = g_type
        self.edcode_type = edcode_type
        self.decode_type = decode_type
        self.nclass = nclass
        self.dorp_code = dorp_code
        self.p_drop = p_drop
        self.drop_en = drop_en
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.drop_edge_rate = drop_edge_rate
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate 
        self.edmask_rate = edmask_rate
        self.cluster_layer = Parameter(torch.Tensor(self.nclass,  output_dim))

        self.loss_type1 = self.setup_loss_fn(loss_fn='sce', alpha_l=3)

        self.loss_type2 = self.setup_loss_fn(loss_fn='sce', alpha_l=3)
        self.loss_type3 = self.setup_loss_fn(loss_fn='bcn')

        self.encode_edge = self.Code(self.edcode_type, self.edin_dim, self.edlatent_dim, self.edout_dim, self.dorp_code)
        self.encoder = Encodeer_Model(self.input_dim, self.en_hidden, self.output_dim, self.p_drop, self.device)
        self.ZINB_decoder = ZINB_decoder(self.zinb_indim, self.zinb_latentdim, self.zinb_outdim)
        self.edge_decoder = EdgeDecoder(self.ed_indim, self.ed_latentdim, self.ed_outdim)
        self.pool_mlp = pool_mlp(self.pm_indim, self.pm_latentdim, self.pm_outdim, self.adj_dim, self.num_layers)
        self.encode_latent = self.Code(self.g_type, self.el_indim, self.el_latentdim, self.el_outdim, self.dorp_code)
        self.decoder = self.Code(self.decode_type, self.dein_dim, self.deout_dim, self.input_dim, self.dorp_code)
        self.decoder1 = InnerProductDecoder(self.p_drop, act=lambda x: x)
 
        self.encode_generate = self.Code(self.eg_type, self.eg_indim, self.eg_latentdim, self.eg_outdim, self.dorp_code)
        self.projector_generate = nn.Sequential(nn.Linear(self.p_latent, self.train_dim),
                                                nn.PReLU(), nn.Linear(self.train_dim, self.p_latent))
        self.projector = nn.Sequential(nn.Linear(self.p_latent, self.train_dim),
                                       nn.PReLU(), nn.Linear(self.train_dim, self.p_latent))
        self.predictor = nn.Sequential(nn.PReLU(), nn.Linear(self.p_latent, self.p_latent))
        self.R = Readout(self.nclass)


        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim)).to(self.device)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.mask_dim)).to(self.device)


    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        

    def forward(self, X, adj, edge_index):
        # feature matrix
        adj, Xmask, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, X, self.mask_rate)
        Zmask = self.encoder(Xmask)
        Xgau = self.gaussian_noised_feature(X)
        Zgau = self.encoder(Xgau)

        #adjacency matrix
        edge_drop = dropout_edge(edge_index, self.drop_edge_rate)  
        num_nodes = adj.shape[0]
        adj_drop = edge_index_to_sparse_adj(edge_drop, num_nodes, device='cuda')
        adj_diff = utils.diffusion_adj(adj, mode="ppr", transport_rate=self.beta)
        # print("adj_diff", adj_diff)
        # print("adj_drop", adj_drop)

        # embedding learning
        Gf1 = self.encode_edge(Xmask, adj_drop)
        Gf2 = self.encode_edge(Xgau, adj_diff)
        H = self.encode_latent(Zmask, adj)
        # H2 = self.encode_latent(Zgau, adj)

        # embedding fusing
        emb = torch.cat([H, Gf1, Gf2], dim=1).to(self.device)
        linear = nn.Linear(self.emb_dim, self.output_dim).to(self.device)
        emb = linear(emb).to(self.device)
        loss_icr = dicr_loss(Gf1, Gf2)

        # rec_loss
        H = H.clone()
        H_rec, _, _ = self.random_remask(adj, H, self.remask_rate) 
        rec = self.decoder(H_rec, adj) 
        x_init = X[mask_nodes]
        x_rec = rec[mask_nodes]
        loss_rec = self.loss_type1(x_init, x_rec)

        #adj_rec
        adj_rec1 = self.decoder1(Gf1, adj)  
        adj_rec2 = self.decoder1(Gf2, adj)
        adj_raw = adj.coalesce().values()
        loss_adj1 = self.loss_type1(adj_raw, adj_rec1)
        loss_adj2 = self.loss_type1(adj_raw, adj_rec2)
        loss_adj = loss_adj1 + loss_adj2

        #KL
        q = 1.0 / ((1.0 + torch.sum((emb.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)  

        return emb, rec, q, loss_rec, loss_de, loss_adj, loss_icr

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            loss_type = nn.MSELoss()
        elif loss_fn == "zinb":
            loss_type = partial(zinb_loss)
        elif loss_fn == 'local_global':
             loss_type = partial(local_global_loss)
        elif loss_fn == 'bcn':
             loss_type = partial(bcn_loss)
        elif loss_fn == "sce":
            loss_type = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return loss_type

    def random_remask(self, adj, rep, remask_rate=0.5):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]
        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token
        return rep, remask_nodes, rekeep_nodes

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()
        return use_adj, out_x, (mask_nodes, keep_nodes)

    def gaussian_noised_feature(self, x):
        N_1 = torch.Tensor(np.random.normal(1, 0.1, x.shape)).to(self.device)
        X_tilde1 = x * N_1
        return X_tilde1


    def mask_edge(self, edge_index, edmaask_rate = 0.7):
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, edmaask_rate, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)  

        remaining_edges = edge_index[:, ~mask]  
        masked_edges = edge_index[:, mask]  

        return remaining_edges, masked_edges


    def Code(self, m_type, in_dim, num_hidden, out_dim, dropout) -> nn.Module:
        if m_type == "GCN":
            mod = GCN(in_dim, num_hidden, out_dim, dropout)
        elif m_type == "mlp":
            mod = nn.Sequential(nn.Linear(in_dim, num_hidden * 2), nn.PReLU(), nn.Dropout(0.2), nn.Linear(num_hidden * 2, out_dim))
        elif m_type == "linear":
            mod = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError
        return mod


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class Encodeer_Model(nn.Module):
    def __init__(self, input_dim, intermediate_dim, kan_dim, p_drop, device):
        super(Encodeer_Model, self).__init__()
        self.device = device
        self.full_block = full_block(input_dim, intermediate_dim,  p_drop).to(self.device)
        self.KAN = KANLinear(intermediate_dim, kan_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.full_block(x)
        feat = self.KAN(x)
        return feat


def dropout_edge(edge_index, p=0.5, force_undirected=False):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return edge_index


class pool_mlp(nn.Module):
    def __init__(self, input_dim, latent_dim, out_dim, adj_dim, num_layers):
        super(pool_mlp, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        self.adj_dim = adj_dim
        # print("adj_dim", adj_dim)
        self.layers.append(layers.GraphConvolution(input_dim, latent_dim).cuda())
        for __ in range(num_layers - 1):
            self.layers.append(layers.GraphConvolution(latent_dim, out_dim).cuda())
        self.ffn1 = nn.Sequential(
            nn.Linear(out_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, out_dim),
            nn.PReLU()
        )
        self.linear_shortcut1 = nn.Linear(out_dim, out_dim)

        self.ffn2 = nn.Sequential(
            nn.Linear(self.num_layers * self.adj_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, out_dim),
            nn.PReLU()
        )
        self.linear_shortcut2 = nn.Linear(self.num_layers * self.adj_dim, out_dim)


    def forward(self, feat, adj):
        Gf = self.layers[0](feat, adj)
        # print("Gf", Gf.size())
        Gf_pool = torch.sum(Gf, 1)
        # print("Gf_pool", Gf_pool.size())
        for idx in range(self.num_layers - 1):
            Gf = self.layers[idx + 1](Gf, adj)
            Gf_pool = torch.cat((Gf_pool, torch.sum(Gf, 1)), -1)
        # print("Gf1", Gf.size())
        # print("Gf_pool1", Gf_pool.size())
        Gf_mlp = self.ffn1(Gf) + self.linear_shortcut1(Gf)
        Gf_pm = self.ffn2(Gf_pool) + self.linear_shortcut2(Gf_pool)
        # print("Gf_mlp", Gf_mlp.size())
        # print("Gf_pm", Gf_pm.size())
        return Gf, Gf_pool, Gf_mlp, Gf_pm


def edge_index_to_sparse_adj(edge_index, num_nodes, device='cuda'):
    edge_weight = torch.ones(edge_index.size(1), device=device) 
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    )
    return adj


class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        # calculate cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        # the cluster-level embedding
        Z_tilde = torch.cat(Z_tilde, dim=0)
        return Z_tilde.view(1, -1)


def dicr_loss1(Z_ae, Z_igae):
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix
    S_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    S_N_igae = cross_correlation(Z_igae[0], Z_igae[1])
    # loss of SCR
    L_N_ae = correlation_reduction_loss(S_N_ae)
    L_N_igae = correlation_reduction_loss(S_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    S_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    S_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(S_F_ae)
    L_F_igae = correlation_reduction_loss(S_F_igae)

    L_N = 0.1 * L_N_ae + 5 * L_N_igae
    L_F = L_F_ae + L_F_igae

    # loss of DICR
    loss_dicr = L_N + L_F

    return loss_dicr


def dicr_loss(com1, com2):

    S_F_ae = cross_correlation(com1.t(), com2.t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(S_F_ae)
    # loss of DICR
    loss_dicr =  L_F_ae

    return loss_dicr


def cross_correlation(Z_v1, Z_v2):

    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):

    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, adj):
        if not adj.is_sparse:
            adj = adj.to_sparse_coo()

        col = adj.coalesce().indices()[0]
        row = adj.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))
        return result

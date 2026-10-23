import copy
import torch
from torch import nn
TOTAL_HIDDEN = 128
PERIOD = 24
LEAF_D_MODEL = 512
LEAF_HEADS = 4
LEAF_ATTN_DROPOUT = 0.5
LEAF_OUT_DROPOUT = 0.0
LEAF_SFM_HIDDEN = 8
LEAF_SFM_KERNEL = 5
LEAF_SFM_GATE = 'raw'
LEAF_SFM_AXIS = 1
LEAF_USE_SFM = True
LEAF_USE_CALENDAR = False
FUSION = 'learned'
MLP_AUX_WEIGHT = 0.5
LEAF_AUX_WEIGHT = 0.5
LOSS_SCALES = 'base'
LOSS_LEVELS = 'level'

def _descendant_matrix(adj):
    adj = torch.as_tensor(adj, dtype=torch.float64)
    n = adj.shape[0]
    desc = torch.eye(n, dtype=torch.float64)
    power = torch.eye(n, dtype=torch.float64)
    for _ in range(n):
        power = power @ adj
        if not power.any():
            return desc
        desc = desc + power
    raise ValueError('Hierarchy contains a cycle.')

def _node_depths(adj):
    adj = torch.as_tensor(adj, dtype=torch.float32)
    n = adj.shape[0]
    depth = torch.full((n,), -1, dtype=torch.long)
    frontier = adj.sum(0) == 0
    level = 0
    while frontier.any():
        if level > n:
            raise ValueError('Hierarchy contains a cycle.')
        depth[frontier] = level
        frontier = adj.t() @ frontier.float() > 0
        level += 1
    if (depth < 0).any():
        raise ValueError('Hierarchy contains a cycle.')
    return depth

def _elementwise(criterion):
    if getattr(criterion, 'reduction', None) == 'none':
        return criterion
    if hasattr(criterion, 'reduction'):
        crit = copy.copy(criterion)
        crit.reduction = 'none'
        return crit
    raise ValueError('criterion must be a torch loss module with a reduction attribute.')

def _seasonal_weight(seq_len, pred_len, period):
    K = max(1, seq_len // period)
    w = torch.zeros(pred_len, seq_len)
    for p in range(pred_len):
        for k in range(1, K + 1):
            idx = seq_len - k * period + p % period
            if 0 <= idx < seq_len:
                w[p, idx] += 1.0 / K
    return w

class HierarchyReconciler(nn.Module):

    def __init__(self, adj, scaler_mean, scaler_scale):
        super().__init__()
        adj = torch.as_tensor(adj, dtype=torch.float32)
        mean = torch.as_tensor(scaler_mean, dtype=torch.float32).flatten()
        scale = torch.as_tensor(scaler_scale, dtype=torch.float32).flatten().clamp_min(1e-08)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        leaf_idx = torch.where(adj.sum(1) == 0)[0]
        S = _descendant_matrix(adj)[:, leaf_idx]
        v = 1.0 / scale.double() ** 2
        St_V = S.t() * v
        self.register_buffer('proj', (S @ torch.linalg.solve(St_V @ S, St_V)).float())

    def forward(self, pred):
        raw = pred * self.scale + self.mean
        raw = torch.einsum('ij,btj->bti', self.proj, raw)
        return (raw - self.mean) / self.scale

class AlignedMultiScaleLoss(nn.Module):

    def __init__(self, adj, factors, scaler_mean, scaler_scale, time_means, time_scales, node_mask=None, scales=LOSS_SCALES, levels=LOSS_LEVELS):
        super().__init__()
        if scales not in ('all', 'base') or levels not in ('level', 'node'):
            raise ValueError(f'Unknown loss mode: scales={scales}, levels={levels}')
        self.factors = list(factors)
        k = len(self.factors)
        self.active = list(range(k)) if scales == 'all' else [0]
        self.register_buffer('mean', torch.as_tensor(scaler_mean, dtype=torch.float32).flatten())
        self.register_buffer('scale', torch.as_tensor(scaler_scale, dtype=torch.float32).flatten().clamp_min(1e-08))
        self.register_buffer('time_means', torch.as_tensor(time_means, dtype=torch.float32).reshape(k, -1))
        self.register_buffer('time_scales', torch.as_tensor(time_scales, dtype=torch.float32).reshape(k, -1).clamp_min(1e-08))
        if node_mask is None and levels == 'node':
            n_nodes = torch.as_tensor(adj).shape[0]
            weights = torch.full((n_nodes,), 1.0 / n_nodes)
        elif node_mask is None:
            depth = _node_depths(adj)
            n_levels = int(depth.max()) + 1
            weights = torch.zeros(len(depth))
            for level in range(n_levels):
                mask = depth == level
                weights[mask] = 1.0 / (float(mask.sum()) * n_levels)
        else:
            mask = torch.as_tensor(node_mask, dtype=torch.bool)
            weights = mask.float() / mask.sum().clamp_min(1)
        self.register_buffer('node_weights', weights)

    def forward(self, pred, target, criterion):
        criterion = _elementwise(criterion)
        raw_p = pred * self.scale + self.mean
        raw_t = target * self.scale + self.mean
        B, T, N = raw_p.shape
        losses = []
        for i in self.active:
            f = self.factors[i]
            p = raw_p.reshape(B, T // f, f, N).sum(2) if f > 1 else raw_p
            t = raw_t.reshape(B, T // f, f, N).sum(2) if f > 1 else raw_t
            p = (p - self.time_means[i]) / self.time_scales[i]
            t = (t - self.time_means[i]) / self.time_scales[i]
            per_node = criterion(p, t).mean(dim=(0, 1))
            losses.append((per_node * self.node_weights).sum())
        return torch.stack(losses).mean()

class AnchorProjection(nn.Module):

    def __init__(self, adj, scaler_mean, scaler_scale):
        super().__init__()
        adj = torch.as_tensor(adj, dtype=torch.float32)
        leaf = adj.sum(1) == 0
        S = _descendant_matrix(adj)[:, leaf]
        C = S[~leaf]
        scale = torch.as_tensor(scaler_scale, dtype=torch.float64).flatten().clamp_min(1e-08)
        w = scale[leaf] ** 2
        gain = (C * w).t() @ torch.linalg.pinv(C * w @ C.t())
        c = float(scale.max())
        self.register_buffer('leaf_mask', leaf)
        self.register_buffer('S', S.float())
        self.register_buffer('C', C.float())
        self.register_buffer('gain', gain.float())
        self.register_buffer('mean', torch.as_tensor(scaler_mean, dtype=torch.float32).flatten() / c)
        self.register_buffer('scale', scale.float() / c)

    def bottom_up(self, leaf_std):
        b = (leaf_std * self.scale + self.mean)[..., self.leaf_mask]
        full = torch.einsum('nl,bpl->bpn', self.S, b)
        return (full - self.mean) / self.scale

    def forward(self, upper_std, leaf_std):
        a = (upper_std * self.scale + self.mean)[..., ~self.leaf_mask]
        b = (leaf_std * self.scale + self.mean)[..., self.leaf_mask]
        resid = a - torch.einsum('ul,bpl->bpu', self.C, b)
        leaves = b + torch.einsum('lu,bpu->bpl', self.gain, resid)
        full = torch.einsum('nl,bpl->bpn', self.S, leaves)
        return (full - self.mean) / self.scale

class TotalBranch(nn.Module):

    def __init__(self, configs, adj):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.enc_in
        self.period = int(getattr(configs, 'period', PERIOD))
        self.cal_dim = int(getattr(configs, 'cal_dim', 0))
        hidden = int(getattr(configs, 'mlp_hidden', TOTAL_HIDDEN))
        dropout = float(configs.dropout)
        L, P, N = (self.seq_len, self.pred_len, self.num_nodes)
        self.reconciler = HierarchyReconciler(adj, configs.scaler_mean, configs.scaler_scale)
        self.loss_fn = AlignedMultiScaleLoss(adj, configs.time_factors, configs.scaler_mean, configs.scaler_scale, configs.time_scaler_means, configs.time_scaler_scales, scales=getattr(configs, 'loss_scales', LOSS_SCALES), levels=getattr(configs, 'loss_levels', LOSS_LEVELS))
        self.shared = nn.Linear(L, P)
        with torch.no_grad():
            self.shared.weight.copy_(_seasonal_weight(L, P, self.period))
            self.shared.bias.zero_()
        emb_dim = 16
        self.node_emb = nn.Embedding(N, emb_dim)
        nn.init.normal_(self.node_emb.weight, std=0.1)
        in_dim = L + P * self.cal_dim + emb_dim
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, P)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, y_mark=None):
        B = x.shape[0]
        level = x.mean(dim=1, keepdim=True)
        xc = (x - level).transpose(1, 2)
        feats = [xc]
        if self.cal_dim > 0:
            feats.append(y_mark.reshape(B, 1, -1).expand(-1, self.num_nodes, -1))
        feats.append(self.node_emb.weight.unsqueeze(0).expand(B, -1, -1))
        out = self.shared(xc) + self.head(self.trunk(torch.cat(feats, dim=-1)))
        return self.reconciler(out.transpose(1, 2) + level)

def _hilbert(x, dim):
    n = x.shape[dim]
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[dim] = n
    return torch.fft.ifft(torch.fft.fft(x, dim=dim) * h.view(shape), dim=dim)

class RevIN(nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def norm(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias

    def denorm(self, y):
        y = (y - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return y * self.stdev + self.mean

class SFM(nn.Module):

    def __init__(self, num_nodes, d_model, hidden, kernel, axis, eps=1e-06, gate=LEAF_SFM_GATE):
        super().__init__()
        if gate not in ('hilbert', 'raw'):
            raise ValueError(f'Unknown SFM gate: {gate}')
        if axis not in (1, 2):
            raise ValueError(f'SFM axis must be 1 (nodes) or 2 (features), got {axis}')
        self.axis = axis
        self.eps = eps
        self.gate = gate
        in_ch = 6 if gate == 'hilbert' else 2
        self.tem = nn.Parameter(torch.zeros(1, num_nodes, d_model))
        self.hilbert_net = nn.Sequential(nn.Conv1d(in_ch, hidden, kernel, padding=kernel // 2), nn.GELU(), nn.Conv1d(hidden, 1, kernel, padding=kernel // 2), nn.GELU())

    def forward(self, x):
        B, N, D = x.shape
        t = self.tem.expand(B, -1, -1)
        if self.gate == 'raw':
            feat = torch.stack([x.detach(), t.detach()], dim=2)
            g = torch.sigmoid(self.hilbert_net(feat.view(B * N, 2, D)).view(B, N, D))
            return x + g * (t - x)
        Zx = _hilbert(x.detach(), self.axis)
        Zt = _hilbert(self.tem.detach(), self.axis).expand(B, -1, -1)
        logAx = torch.log(Zx.abs().clamp_min(self.eps))
        logAt = torch.log(Zt.abs().clamp_min(self.eps))
        phix, phit = (torch.angle(Zx), torch.angle(Zt))
        feat = torch.stack([logAx, logAt, torch.cos(phix), torch.sin(phix), torch.cos(phit), torch.sin(phit)], dim=2)
        g = torch.sigmoid(self.hilbert_net(feat.view(B * N, 6, D)).view(B, N, D))
        return x + g * (t - x)

class LeafBranch(nn.Module):

    def __init__(self, configs, adj):
        super().__init__()
        self.seq_len = L = configs.seq_len
        self.pred_len = P = configs.pred_len
        self.num_nodes = N = configs.enc_in
        D = int(getattr(configs, 'leaf_d_model', LEAF_D_MODEL))
        if D % LEAF_HEADS:
            raise ValueError(f'leaf_d_model must be divisible by {LEAF_HEADS}.')
        cal_dim = int(getattr(configs, 'cal_dim', 0))
        self.use_sfm = bool(getattr(configs, 'leaf_use_sfm', LEAF_USE_SFM))
        self.use_calendar = bool(getattr(configs, 'leaf_use_calendar', LEAF_USE_CALENDAR)) and cal_dim > 0
        self.revin = RevIN(N)
        self.embed = nn.Sequential(nn.Linear(L, D), nn.GELU())
        if self.use_sfm:
            self.sfm = SFM(N, D, LEAF_SFM_HIDDEN, LEAF_SFM_KERNEL, int(getattr(configs, 'leaf_sfm_axis', LEAF_SFM_AXIS)), gate=str(getattr(configs, 'leaf_sfm_gate', LEAF_SFM_GATE)))
        if self.use_calendar:
            self.cal_proj = nn.Linear(P * cal_dim, D)
            nn.init.zeros_(self.cal_proj.weight)
            nn.init.zeros_(self.cal_proj.bias)
        self.attn = nn.MultiheadAttention(D, LEAF_HEADS, dropout=LEAF_ATTN_DROPOUT, batch_first=True)
        self.attn_proj = nn.Sequential(nn.Linear(D, D), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(LEAF_OUT_DROPOUT), nn.Linear(D, P))

    def forward(self, x, y_mark=None):
        B = x.shape[0]
        z = self.revin.norm(x)
        h = self.embed(z.transpose(1, 2))
        if self.use_sfm:
            h = self.sfm(h)
        if self.use_calendar:
            h = h + self.cal_proj(y_mark.reshape(B, 1, -1))
        h = self.attn_proj(self.attn(h, h, h, need_weights=False)[0] + h)
        y = self.out(h).transpose(1, 2)
        return self.revin.denorm(y)

class seasleaf(nn.Module):

    def __init__(self, configs, **kwargs):
        super().__init__()
        adj = torch.as_tensor(configs.adj, dtype=torch.float32)
        for f in configs.time_factors:
            if configs.seq_len % f or configs.pred_len % f:
                raise ValueError('Every time factor must divide seq_len and pred_len.')
        self.fusion = str(getattr(configs, 'fusion', FUSION))
        if self.fusion not in ('anchor', 'total', 'leaf', 'mean', 'learned'):
            raise ValueError(f'Unknown fusion: {self.fusion}')
        self.cal_dim = int(getattr(configs, 'cal_dim', 0))
        self.total_branch = TotalBranch(configs, adj) if self.fusion != 'leaf' else None
        self.leaf_branch = LeafBranch(configs, adj) if self.fusion != 'total' else None
        self.anchor = AnchorProjection(adj, configs.scaler_mean, configs.scaler_scale)
        self.reconciler = HierarchyReconciler(adj, configs.scaler_mean, configs.scaler_scale)
        self.loss_fn = AlignedMultiScaleLoss(adj, configs.time_factors, configs.scaler_mean, configs.scaler_scale, configs.time_scaler_means, configs.time_scaler_scales, scales=getattr(configs, 'loss_scales', LOSS_SCALES), levels=getattr(configs, 'loss_levels', LOSS_LEVELS))
        self.leaf_loss_fn = AlignedMultiScaleLoss(adj, configs.time_factors, configs.scaler_mean, configs.scaler_scale, configs.time_scaler_means, configs.time_scaler_scales, node_mask=adj.sum(1) == 0, scales=getattr(configs, 'loss_scales', LOSS_SCALES))
        if self.fusion == 'learned':
            self.fusion_logit = nn.Parameter(torch.zeros(configs.pred_len, configs.enc_in))
        two_branch = self.fusion in ('anchor', 'mean', 'learned')
        self.mlp_aux_weight = float(getattr(configs, 'mlp_aux_weight', MLP_AUX_WEIGHT)) if two_branch else 0.0
        self.leaf_aux_weight = float(getattr(configs, 'leaf_aux_weight', LEAF_AUX_WEIGHT)) if two_branch else 0.0
        self.last_branch = {}
        self.last_losses = {}

    def forward_branches(self, x, x_mark=None, y_mark=None):
        if self.cal_dim > 0 and (y_mark is None or y_mark.shape[-1] != self.cal_dim):
            raise ValueError(f'Model needs y_mark [B, pred_len, {self.cal_dim}].')
        total = self.total_branch(x, y_mark) if self.total_branch is not None else None
        leaf = self.leaf_branch(x, y_mark) if self.leaf_branch is not None else None
        return (total, leaf)

    def fuse(self, total, leaf):
        if self.fusion == 'anchor':
            return self.anchor(total, leaf)
        if self.fusion == 'total':
            return total
        if self.fusion == 'leaf':
            return self.anchor.bottom_up(leaf)
        if self.fusion == 'learned':
            w = torch.sigmoid(self.fusion_logit)
            return self.reconciler(w * total + (1.0 - w) * leaf)
        return self.reconciler(0.5 * (total + leaf))

    def fusion_weights(self):
        if self.fusion != 'learned':
            return None
        return torch.sigmoid(self.fusion_logit).detach().cpu()

    def forward(self, x, x_mark=None, y_mark=None):
        total, leaf = self.forward_branches(x, x_mark, y_mark)
        self.last_branch = {'total': total, 'leaf': leaf}
        return self.fuse(total, leaf)

    def compute_loss(self, prediction, target, criterion):
        main = self.loss_fn(prediction, target, criterion)
        losses = {'main': main.detach()}
        total = main
        if self.training and self.mlp_aux_weight > 0:
            l_total = self.total_branch.loss_fn(self.last_branch['total'], target, criterion)
            total = total + self.mlp_aux_weight * l_total
            losses['total'] = l_total.detach()
        if self.training and self.leaf_aux_weight > 0:
            l_leaf = self.leaf_loss_fn(self.last_branch['leaf'], target, criterion)
            total = total + self.leaf_aux_weight * l_leaf
            losses['leaf'] = l_leaf.detach()
        self.last_losses = losses
        return total

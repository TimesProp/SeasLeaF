import argparse
from contextlib import redirect_stdout
import json
import hashlib
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.seasleaf import seasleaf
OUTPUT_DIR = ROOT / 'outputs' / 'seasleaf'
VERSION = 'seasleaf'
DATA_PATH = str(ROOT / 'data' / 'mixedhumid_sfd.csv')
HIER_PATH = str(ROOT / 'data' / 'mixedhumid_sfd_hier.csv')
SEQ_LEN = 120
PRED_LEN = 24
TIME_HIERARCHY = [1, 3, 2, 2]
BASE_INTERVAL_MIN = 60
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 100
RUN_TIMES = 3
LR = 0.0005
PATIENCE = 10
SEED = 42
LOSS_NAME = 'mse'
WEIGHT_DECAY = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN_TAG = ''
SMOKE = False
TF32 = False
PROFILE = False
DROPOUT = 0.1
MLP_HIDDEN = 128
USE_CALENDAR = True
LEAF_D_MODEL = 512
LEAF_USE_SFM = True
LEAF_SFM_GATE = 'raw'
LEAF_SFM_AXIS = 1
NODE_PERM_SEED = -1
LEAF_USE_CALENDAR = False
FUSION = 'learned'
LOSS_SCALES = 'base'
LOSS_LEVELS = 'level'
MLP_AUX_WEIGHT = 0.5
LEAF_AUX_WEIGHT = 0.5

class WindowDataset(Dataset):

    def __init__(self, data, marks, start, end, seq_len, pred_len):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.marks = torch.as_tensor(marks, dtype=torch.float32)
        self.start, self.end = (start, end)
        self.seq_len, self.pred_len = (seq_len, pred_len)
        if start < seq_len or end - start < pred_len:
            raise ValueError('Dataset split is too short for seq_len/pred_len.')

    def __len__(self):
        return self.end - self.start - self.pred_len + 1

    def __getitem__(self, index):
        t = self.start + index
        return (self.data[t - self.seq_len:t], self.data[t:t + self.pred_len], self.marks[t - self.seq_len:t], self.marks[t:t + self.pred_len])

class DeviceWindowLoader:

    def __init__(self, dataset, batch_size, shuffle, device, seed=0):
        self.data = dataset.data.to(device)
        self.marks = dataset.marks.to(device)
        self.starts = torch.arange(dataset.start, dataset.end - dataset.pred_len + 1, device=device)
        self.past = torch.arange(-dataset.seq_len, 0, device=device)
        self.future = torch.arange(0, dataset.pred_len, device=device)
        self.batch_size, self.shuffle = (batch_size, shuffle)
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset = dataset

    def __len__(self):
        return (len(self.starts) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.starts)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator).to(self.starts.device)
        else:
            order = torch.arange(n, device=self.starts.device)
        for i in range(0, n, self.batch_size):
            t = self.starts[order[i:i + self.batch_size]].unsqueeze(1)
            past, future = (t + self.past, t + self.future)
            yield (self.data[past], self.data[future], self.marks[past], self.marks[future])

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def aggregate(x, factor):
    if factor == 1:
        return x
    T = x.shape[-2] // factor * factor
    if x.ndim == 2:
        return x[:T].reshape(-1, factor, x.shape[-1]).sum(axis=1)
    return x[:, :T].reshape(x.shape[0], -1, factor, x.shape[-1]).sum(axis=2)

def build_multiscale(raw, time_hierarchy):
    levels = [raw]
    current = raw
    for ratio in time_hierarchy[1:]:
        current = aggregate(current, ratio)
        levels.append(current)
    return levels

def build_scale_factors(time_hierarchy):
    factors = []
    cumulative = 1
    for ratio in time_hierarchy:
        cumulative *= ratio
        factors.append(cumulative)
    return factors

def format_interval(minutes):
    if minutes >= 1440 and minutes % 1440 == 0:
        days = minutes // 1440
        return f'{days}day' if days == 1 else f'{days}days'
    if minutes < 60:
        return f'{minutes}min'
    if minutes % 60 == 0:
        hours = minutes // 60
        return f'{hours}hour' if hours == 1 else f'{hours}hours'
    return f'{minutes}min'

def get_stats(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-06] = 1.0
    return (mean, std)

def bottom_up_reconcile(pred, node_names, children, node_levels):
    result = pred.copy()
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    max_level = node_levels.max()
    for level in range(max_level - 1, -1, -1):
        for i, node in enumerate(node_names):
            if node_levels[i] == level and len(children[node]) > 0:
                child_idx = [name_to_idx[c] for c in children[node]]
                result[:, :, i] = result[:, :, child_idx].sum(axis=2)
    return result

def top_down_reconcile(pred, node_names, children, node_levels):
    result = pred.copy()
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    max_level = node_levels.max()
    eps = 1e-08
    for level in range(max_level):
        for i, parent in enumerate(node_names):
            if node_levels[i] != level or len(children[parent]) == 0:
                continue
            child_idx = [name_to_idx[c] for c in children[parent]]
            original_children = pred[:, :, child_idx]
            child_sum = original_children.sum(axis=2, keepdims=True)
            denom = np.where(np.abs(child_sum) < eps, 1.0, child_sum)
            proportions = original_children / denom
            zero_mask = np.abs(child_sum) < eps
            proportions = np.where(zero_mask, 1.0 / len(child_idx), proportions)
            result[:, :, child_idx] = result[:, :, i:i + 1] * proportions
    return result

def format_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return np.char.add(np.char.add(np.char.mod('%.6f', mean), '±'), np.char.mod('%.6f', std))

def print_summary(results):
    print(f'\nCompleted runs: {len(results)} | Sample std (ddof=1); single run std=0')
    print('========== TEST ==========')
    for metric in results[0]['metrics']:
        value = format_mean_std([r['metrics'][metric] for r in results])
        print(f'{metric}: {value.item()}')
    tables = {metric: np.stack([r['tables'][metric] for r in results]) for metric in results[0]['tables']}
    scales = results[0]['scale_names']
    levels = results[0]['data_level_names']
    for metric, values in tables.items():
        print(f'\n========== {metric} ==========')
        print('\t'.join(['Time'] + levels))
        for name, row in zip(scales, format_mean_std(values)):
            print('\t'.join([name] + row.tolist()))
    for title, names, axis in [('AVERAGE ACROSS TIME LEVELS', levels, 1), ('AVERAGE ACROSS DATA LEVELS', scales, 2)]:
        print(f'\n========== {title} ==========')
        print('\t'.join(['Metric'] + names))
        for metric, values in tables.items():
            row = format_mean_std(values.mean(axis=axis))
            print('\t'.join([metric] + row.tolist()))

def build_calendar_features(time_col, n, base_interval_min):
    ts = None
    if not pd.api.types.is_numeric_dtype(time_col):
        parsed = pd.to_datetime(time_col, errors='coerce')
        if not parsed.isna().any() and parsed.is_monotonic_increasing:
            ts = pd.DatetimeIndex(parsed)
    if ts is None:
        print('[WARN] 时间列无法解析为递增时间戳，改为假设数据从周一 00:00 开始、等间隔采样来生成日历特征。', flush=True)
        ts = pd.DatetimeIndex(pd.Timestamp('2000-01-03') + pd.to_timedelta(np.arange(n) * base_interval_min, unit='m'))
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    feats = []
    for k in (1, 2, 3):
        angle = 2 * np.pi * k * hour / 24.0
        feats += [np.sin(angle), np.cos(angle)]
    angle = 2 * np.pi * ts.dayofweek.to_numpy() / 7.0
    feats += [np.sin(angle), np.cos(angle)]
    angle = 2 * np.pi * (ts.month.to_numpy() - 1) / 12.0
    feats += [np.sin(angle), np.cos(angle)]
    return np.stack(feats, axis=1).astype(np.float32)

def load_data():
    frame = pd.read_csv(DATA_PATH)
    if SMOKE:
        frame = frame.head(360)
    names = frame.columns[1:].tolist()
    data = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not names or not np.isfinite(data).all():
        raise ValueError('Data must contain a time column and finite numeric node columns.')
    if NODE_PERM_SEED >= 0:
        perm = np.random.default_rng(NODE_PERM_SEED).permutation(len(names))
        names = [names[i] for i in perm]
        data = data[:, perm]
        print(f'[PERM] node order shuffled with seed {NODE_PERM_SEED}: {names}', flush=True)
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    if train_end < SEQ_LEN + PRED_LEN or val_end - train_end < PRED_LEN or n - val_end < PRED_LEN:
        raise ValueError('Data splits are too short; increase data length or reduce window.')
    scaler = StandardScaler().fit(data[:train_end])
    scaled = scaler.transform(data)
    stats = [get_stats(level) for level in build_multiscale(data[:train_end], TIME_HIERARCHY)]
    hierarchy = pd.read_csv(HIER_PATH, index_col=0)
    if hierarchy.index.has_duplicates or hierarchy.columns.has_duplicates:
        raise ValueError('Hierarchy node names must be unique.')
    missing = [node for node in names if node not in hierarchy.index or node not in hierarchy.columns]
    if missing:
        raise ValueError(f'Nodes missing in hierarchy: {missing}')
    adj = hierarchy.loc[names, names].to_numpy(dtype=np.float32)
    if not np.isfinite(adj).all() or not np.isin(adj, [0, 1]).all() or (adj.sum(0) > 1).any():
        raise ValueError('Hierarchy must be a binary tree/forest adjacency.')
    levels = np.full(len(names), -1, dtype=int)
    queue = list(np.where(adj.sum(0) == 0)[0])
    levels[queue] = 0
    for parent in queue:
        for child in np.where(adj[parent] > 0)[0]:
            levels[child] = levels[parent] + 1
            queue.append(int(child))
    if len(queue) != len(names):
        raise ValueError('Hierarchy contains a cycle.')
    children = {node: [names[j] for j in np.where(adj[i] > 0)[0]] for i, node in enumerate(names)}
    marks = build_calendar_features(frame.iloc[:, 0], n, BASE_INTERVAL_MIN)
    parent_mask = adj.sum(1) > 0
    if parent_mask.any():
        child_sum = data @ adj.T
        gap = np.abs(data[:, parent_mask] - child_sum[:, parent_mask]).mean(0)
        rel = gap / (np.abs(data[:, parent_mask]).mean(0) + 1e-08)
        print(f'[CHECK] 数据层级一致性：父节点与子节点之和的相对误差 max={rel.max():.3e} mean={rel.mean():.3e}' + ('  <-- 数据并不严格相干，BU/TD/投影可能引入偏差' if rel.max() > 0.001 else ''), flush=True)
    datasets = [WindowDataset(scaled, marks, start, end, SEQ_LEN, PRED_LEN) for start, end in [(SEQ_LEN, train_end), (train_end, val_end), (val_end, n)]]
    return (datasets, scaler, stats, names, adj, children, levels)

def build_config(names, adj, scaler, stats, cal_dim):
    return SimpleNamespace(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=len(names), time_factors=build_scale_factors(TIME_HIERARCHY), adj=adj, dropout=DROPOUT, mlp_hidden=MLP_HIDDEN, leaf_d_model=LEAF_D_MODEL, leaf_use_sfm=LEAF_USE_SFM, leaf_sfm_axis=LEAF_SFM_AXIS, leaf_sfm_gate=LEAF_SFM_GATE, leaf_use_calendar=LEAF_USE_CALENDAR, fusion=FUSION, loss_scales=LOSS_SCALES, loss_levels=LOSS_LEVELS, mlp_aux_weight=MLP_AUX_WEIGHT, leaf_aux_weight=LEAF_AUX_WEIGHT, cal_dim=cal_dim, period=max(1, int(round(24 * 60 / BASE_INTERVAL_MIN))), scaler_mean=scaler.mean_.astype(np.float32), scaler_scale=scaler.scale_.astype(np.float32), time_scaler_means=np.concatenate([mean for mean, _ in stats], axis=0).astype(np.float32), time_scaler_scales=np.concatenate([std for _, std in stats], axis=0).astype(np.float32))

def predict(model, batch, device):
    x, y, x_mark, y_mark = [t.to(device) for t in batch]
    return (model(x, x_mark, y_mark), y)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, count = (0.0, 0)
    for batch in loader:
        pred, target = predict(model, batch, device)
        total += model.compute_loss(pred, target, criterion) * len(pred)
        count += len(pred)
    return float(total) / count

def make_criterion():
    return {'huber': nn.HuberLoss, 'mse': nn.MSELoss, 'mae': nn.L1Loss}[LOSS_NAME](reduction='none')

def train_model(model, loaders, device, checkpoint):
    criterion = make_criterion()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(1, PATIENCE // 3))
    best, wait = (float('inf'), 0)
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_total = torch.zeros((), device=device)
        train_count = 0
        for batch in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            pred, target = predict(model, batch, device)
            loss = model.compute_loss(pred, target, criterion)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss in epoch {epoch}.')
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_total += loss.detach() * len(pred)
            train_count += len(pred)
        train_time = time.perf_counter() - epoch_started
        train_loss = train_total.item() / train_count
        val = validate(model, loaders[1], criterion, device)
        if not np.isfinite(val):
            raise RuntimeError('Non-finite validation loss.')
        scheduler.step(val)
        print(f"[FAST] Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val:.6f} | LR {optimizer.param_groups[0]['lr']:.1e} | {time.perf_counter() - epoch_started:.2f}s (train {train_time:.2f}s)", flush=True)
        if val < best:
            best, wait = (val, 0)
            torch.save(model.state_dict(), checkpoint)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f'Stop at Epoch: {epoch}')
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

@torch.no_grad()
def evaluate(model, loader, device, scaler, stats, names, children, levels):
    model.eval()
    predictions, targets = ([], [])
    for batch in loader:
        pred, target = predict(model, batch, device)
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    pred = np.concatenate(predictions)
    true = np.concatenate(targets)
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite test predictions.')
    shape = pred.shape
    raw_pred = scaler.inverse_transform(pred.reshape(-1, len(names))).reshape(shape)
    raw_true = scaler.inverse_transform(true.reshape(-1, len(names))).reshape(shape)
    errors = raw_pred - raw_true
    metrics = {'Normalized MSE': np.mean((pred - true) ** 2), 'Normalized MAE': np.mean(abs(pred - true)), 'Real-scale MSE': np.mean(errors ** 2), 'Real-scale RMSE': np.sqrt(np.mean(errors ** 2)), 'Real-scale MAE': np.mean(abs(errors))}
    scales = build_scale_factors(TIME_HIERARCHY)
    true_levels = build_multiscale(raw_true, TIME_HIERARCHY)
    res = {}
    for label, reconcile in [('BU', bottom_up_reconcile), ('TD', top_down_reconcile)]:
        reconciled_pred = reconcile(raw_pred, names, children, levels)
        pred_levels = build_multiscale(reconciled_pred, TIME_HIERARCHY)
        mse = np.zeros((len(scales), levels.max() + 1))
        mae = np.zeros_like(mse)
        for i, (p, t, (mean, std)) in enumerate(zip(pred_levels, true_levels, stats)):
            error = (p - mean) / std - (t - mean) / std
            for level in range(levels.max() + 1):
                e = error[..., levels == level]
                mse[i, level] = np.mean(e ** 2)
                mae[i, level] = np.mean(abs(e))
        res[label + ' MSE'], res[label + ' MAE'] = (mse, mae)
    tables = {'BU MSE': res['BU MSE'], 'TD MSE': res['TD MSE'], 'BU MAE': res['BU MAE'], 'TD MAE': res['TD MAE']}
    return dict(metrics=metrics, tables=tables, scale_names=[format_interval(BASE_INTERVAL_MIN * f) for f in scales], data_level_names=[f'L{i}' for i in range(levels.max() + 1)])

def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def parse_time_hierarchy(values):
    if values is None:
        return list(TIME_HIERARCHY)
    joined = ' '.join(values).strip()
    if joined.startswith('[') and joined.endswith(']'):
        joined = joined[1:-1]
    parts = joined.replace(',', ' ').split()
    if not parts:
        raise argparse.ArgumentTypeError('time_hierarchy cannot be empty.')
    try:
        ratios = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('time_hierarchy must contain integers, e.g. 1 2 3.') from exc
    if ratios[0] != 1 or any((ratio < 1 for ratio in ratios)):
        raise argparse.ArgumentTypeError('time_hierarchy must start with 1 and all ratios must be positive.')
    return ratios

def profile_model(model, loader, device, steps=20):
    criterion = make_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    sync = torch.cuda.synchronize if device.type == 'cuda' else lambda: None
    names = ['数据切片', '两个分支前向', '锚定层', '损失(含辅助损失)', '反向+优化器']
    cost = dict.fromkeys(names, 0.0)

    def tick():
        sync()
        return time.perf_counter()
    model.train()
    it = iter(loader)
    for step in range(steps + 3):
        t0 = tick()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y, xm, ym = [b.to(device) for b in batch]
        t1 = tick()
        total, leaf = model.forward_branches(x, xm, ym)
        t2 = tick()
        model.last_branch = {'total': total, 'leaf': leaf}
        pred = model.fuse(total, leaf)
        t3 = tick()
        loss = model.compute_loss(pred, y, criterion)
        t4 = tick()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        t5 = tick()
        if step >= 3:
            for k, v in zip(names, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]):
                cost[k] += v
    per = {k: 1000 * v / steps for k, v in cost.items()}
    step_ms = sum(per.values())
    n_params = sum((p.numel() for p in model.parameters()))
    leaf_params = sum((p.numel() for p in model.leaf_branch.parameters())) if model.leaf_branch is not None else 0
    print(f'\n===== 耗时分析 [fusion={FUSION}]: 每个 batch 平均 (batch={BATCH_SIZE}, {steps} 步) =====')
    for k, v in per.items():
        print(f'{k:24s}{v:8.1f} ms')
    print(f"{'合计':24s}{step_ms:8.1f} ms  -> 每轮训练约 {step_ms * len(loader) / 1000:.1f}s ({len(loader)} 个 batch)")
    print(f'参数量: 总 {n_params:,} | 叶子分支 {leaf_params:,}')

def model_tag():
    tag = 'seasleaf'
    changes = [(not USE_CALENDAR, 'nocal'), (not LEAF_USE_SFM, 'nosfm'), (LEAF_USE_SFM and LEAF_SFM_GATE != 'raw', f'sfmgate-{LEAF_SFM_GATE}'), (LEAF_USE_SFM and LEAF_SFM_GATE == 'hilbert' and (LEAF_SFM_AXIS != 1), f'sfmaxis{LEAF_SFM_AXIS}'), (NODE_PERM_SEED >= 0, f'perm{NODE_PERM_SEED}'), (LEAF_USE_CALENDAR, 'leafcal'), (FUSION != 'learned', f'fusion-{FUSION}'), (LOSS_SCALES != 'base', f'loss-{LOSS_SCALES}'), (LOSS_LEVELS != 'level', f'loss-{LOSS_LEVELS}'), (MLP_AUX_WEIGHT != 0.5, f'mlpaux{MLP_AUX_WEIGHT:g}'), (LEAF_AUX_WEIGHT != 0.5, f'leafaux{LEAF_AUX_WEIGHT:g}'), (MLP_HIDDEN != 128, f'mlp{MLP_HIDDEN}'), (LEAF_D_MODEL != 512, f'd{LEAF_D_MODEL}')]
    for changed, name in changes:
        if changed:
            tag += '_' + name
    return tag

def main():
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = TF32
        torch.backends.cudnn.allow_tf32 = TF32
    tag = model_tag()
    datasets, scaler, stats, names, adj, children, levels = load_data()
    cal_dim = datasets[0].marks.shape[1] if USE_CALENDAR else 0
    config = build_config(names, adj, scaler, stats, cal_dim)
    data_stem = Path(DATA_PATH).stem
    folder = OUTPUT_DIR / ('smoke' if SMOKE else 'experiments') / VERSION / tag / data_stem / ('time_hierarchy_' + '_'.join(map(str, TIME_HIERARCHY)))
    folder /= f'seq_{SEQ_LEN}_pred_{PRED_LEN}'
    if RUN_TAG:
        folder /= RUN_TAG
    folder.mkdir(parents=True, exist_ok=True)
    scale_factors = build_scale_factors(TIME_HIERARCHY)
    scale_names = [format_interval(BASE_INTERVAL_MIN * f) for f in scale_factors]
    print(f'Model | {tag} | device={DEVICE} | output={folder}', flush=True)
    print('SEQ_LEN:', SEQ_LEN, '| PRED_LEN:', PRED_LEN)
    print('TIME_HIERARCHY:', TIME_HIERARCHY, '| FACTORS:', scale_factors, '| SCALES:', scale_names)
    results = []
    fusion_weight_log = []
    for run in range(RUN_TIMES):
        seed = SEED + run
        seed_everything(seed)
        loaders = [DeviceWindowLoader(ds, BATCH_SIZE, shuffle=i == 0, device=DEVICE, seed=seed) for i, ds in enumerate(datasets)]
        model = seasleaf(config).to(DEVICE)
        param_count = sum((p.numel() for p in model.parameters()))
        print(f'\n========== RUN {run + 1}/{RUN_TIMES} | SEED {seed} | Params={param_count} ==========', flush=True)
        if PROFILE:
            profile_model(model, loaders[0], DEVICE)
            return
        metadata = dict(version=VERSION, tag=tag, seq_len=SEQ_LEN, pred_len=PRED_LEN, data_path=DATA_PATH, hier_path=HIER_PATH, hierarchy_config=vars(config), node_names=names, seed=seed)
        metadata['comparison_protocol'] = dict(data_sha256=hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest(), hierarchy_sha256=hashlib.sha256(Path(HIER_PATH).read_bytes()).hexdigest(), seq_len=SEQ_LEN, pred_len=PRED_LEN, time_hierarchy=TIME_HIERARCHY, base_interval_min=BASE_INTERVAL_MIN, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, batch_size=BATCH_SIZE, epochs=EPOCHS, runs=RUN_TIMES, seed_start=SEED, optimizer='AdamW', lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE, scheduler='ReduceLROnPlateau/factor=0.5/patience=max(1,patience//3)', loss=LOSS_NAME, loss_scales=LOSS_SCALES, loss_levels=LOSS_LEVELS, tf32=TF32, smoke=SMOKE, node_names=names, node_perm_seed=NODE_PERM_SEED, training_protocol='seasleaf')
        (folder / f'config_seed{seed}.json').write_text(json.dumps(metadata, default=json_value, indent=2), encoding='utf-8')
        train_model(model, loaders, DEVICE, folder / f'best_seed{seed}.pt')
        results.append(evaluate(model, loaders[2], DEVICE, scaler, stats, names, children, levels))
        (folder / 'results.json').write_text(json.dumps(results, default=json_value, indent=2), encoding='utf-8')
        weights = model.fusion_weights()
        if weights is not None:
            w = weights.numpy()
            np.save(folder / f'fusion_weights_seed{seed}.npy', w)
            level_w = [float(w[:, levels == lv].mean()) for lv in range(levels.max() + 1)]
            fusion_weight_log.append(level_w)
            print('Learned fusion weight on total branch (mean per level): ' + ' | '.join((f'L{lv}={v:.3f}' for lv, v in enumerate(level_w))), flush=True)
        del model, loaders
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print_summary(results)
    temp_path = ROOT / ('smoke_temp.txt' if SMOKE else 'temp.txt')
    with temp_path.open('a', encoding='utf-8') as temp_file:
        with redirect_stdout(temp_file):
            print_summary(results)
            print('Model: seasleaf')
            print(f'Base: {tag}')
            print(f'Version: {VERSION}')
            print(f'TOTAL: hidden={MLP_HIDDEN} | dropout={DROPOUT} | calendar={USE_CALENDAR}')
            print(f'LEAF: d_model={LEAF_D_MODEL} | sfm={LEAF_USE_SFM} | sfm_axis={LEAF_SFM_AXIS} | sfm_gate={LEAF_SFM_GATE} | leaf_calendar={LEAF_USE_CALENDAR}')
            print(f"NODE ORDER: {('csv' if NODE_PERM_SEED < 0 else f'shuffled (seed {NODE_PERM_SEED})')}")
            print(f'FUSION: {FUSION} | LOSS: scales={LOSS_SCALES}, levels={LOSS_LEVELS}')
            if fusion_weight_log:
                arr = np.asarray(fusion_weight_log)
                print('FUSION WEIGHT on total branch (mean over seeds, per level): ' + ' | '.join((f'L{lv}={arr[:, lv].mean():.3f}±{arr[:, lv].std():.3f}' for lv in range(arr.shape[1]))))
            print(f'AUX: total={MLP_AUX_WEIGHT} | leaf={LEAF_AUX_WEIGHT}')
            print(f'Data: {data_stem}')
            print(f'LR: {LR} | WD: {WEIGHT_DECAY} | LOSS: {LOSS_NAME}')
            print(f'Lookback Window: {SEQ_LEN}')
            print(f'Prediction Window: {PRED_LEN}')
            print(f'Time Hierarchy: {TIME_HIERARCHY}')
            print(f'Time Scales: {scale_names}')
            print('########')
    print(f'\nEvaluation complete. Results appended to: {temp_path}', flush=True)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    g = parser.add_argument_group('数据与训练')
    g.add_argument('--seq', '--seq-len', dest='seq_len', type=int, default=SEQ_LEN)
    g.add_argument('--pred', '--pred-len', '--pred_len', dest='pred_len', type=int, default=PRED_LEN)
    g.add_argument('--path', type=str, default=str(ROOT / 'data' / 'mixedhumid_sfd'))
    g.add_argument('--time_hierarchy', '--time-hierarchy', nargs='+', default=None, metavar='RATIO')
    g.add_argument('--base-interval-min', type=int, default=BASE_INTERVAL_MIN)
    g.add_argument('--epochs', type=int, default=EPOCHS)
    g.add_argument('--runs', type=int, default=RUN_TIMES)
    g.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    g.add_argument('--lr', type=float, default=LR)
    g.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    g.add_argument('--patience', type=int, default=PATIENCE)
    g.add_argument('--seed', type=int, default=SEED)
    g.add_argument('--loss', choices=['huber', 'mse', 'mae'], default=LOSS_NAME)
    g.add_argument('--device', default='auto')
    g.add_argument('--tag', default='')
    g.add_argument('--tf32', action='store_true')
    g.add_argument('--profile', action='store_true')
    g.add_argument('--smoke', action='store_true')
    m = parser.add_argument_group('模型宽度')
    m.add_argument('--mlp-hidden', type=int, default=MLP_HIDDEN)
    m.add_argument('--dropout', type=float, default=DROPOUT)
    m.add_argument('--leaf-d-model', type=int, default=LEAF_D_MODEL)
    a = parser.add_argument_group('消融开关（默认即最终模型）')
    a.add_argument('--fusion', choices=['learned', 'mean', 'anchor', 'total', 'leaf'], default=FUSION)
    a.add_argument('--no-calendar', action='store_true')
    a.add_argument('--leaf-no-sfm', action='store_true')
    a.add_argument('--leaf-sfm-axis', type=int, choices=[1, 2], default=LEAF_SFM_AXIS)
    a.add_argument('--leaf-sfm-gate', choices=['hilbert', 'raw'], default=LEAF_SFM_GATE)
    a.add_argument('--node-perm-seed', type=int, default=NODE_PERM_SEED)
    a.add_argument('--leaf-calendar', action='store_true')
    a.add_argument('--loss-scales', choices=['base', 'all'], default=LOSS_SCALES)
    a.add_argument('--loss-levels', choices=['level', 'node'], default=LOSS_LEVELS)
    a.add_argument('--mlp-aux-weight', type=float, default=MLP_AUX_WEIGHT)
    a.add_argument('--leaf-aux-weight', type=float, default=LEAF_AUX_WEIGHT)
    args = parser.parse_args(argv)
    try:
        args.time_hierarchy = parse_time_hierarchy(args.time_hierarchy)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.seq_len <= 0 or args.pred_len <= 0:
        parser.error('--seq and --pred must both be positive integers.')
    factors = build_scale_factors(args.time_hierarchy)
    if args.seq_len % factors[-1] != 0 or args.pred_len % factors[-1] != 0:
        parser.error(f'Seq len ({args.seq_len}) and pred len ({args.pred_len}) must be divisible by coarsest scale factor ({factors[-1]}).')
    if args.smoke:
        args.epochs = args.runs = 1
    return args

def cli(argv=None):
    global SEQ_LEN, PRED_LEN, TIME_HIERARCHY, BASE_INTERVAL_MIN
    global EPOCHS, RUN_TIMES, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, SEED
    global LOSS_NAME, RUN_TAG, SMOKE, DEVICE, DATA_PATH, HIER_PATH, TF32, PROFILE
    global DROPOUT, MLP_HIDDEN, USE_CALENDAR, LEAF_D_MODEL, LEAF_USE_SFM
    global LEAF_USE_CALENDAR, FUSION, LOSS_SCALES, LOSS_LEVELS
    global MLP_AUX_WEIGHT, LEAF_AUX_WEIGHT
    global LEAF_SFM_AXIS, LEAF_SFM_GATE, NODE_PERM_SEED
    args = parse_args(argv)
    SEQ_LEN, PRED_LEN = (args.seq_len, args.pred_len)
    TIME_HIERARCHY = args.time_hierarchy
    BASE_INTERVAL_MIN = args.base_interval_min
    EPOCHS, RUN_TIMES, BATCH_SIZE = (args.epochs, args.runs, args.batch_size)
    LR, WEIGHT_DECAY, PATIENCE, SEED = (args.lr, args.weight_decay, args.patience, args.seed)
    LOSS_NAME, RUN_TAG, SMOKE = (args.loss, args.tag, args.smoke)
    TF32, PROFILE = (args.tf32, args.profile)
    DROPOUT, MLP_HIDDEN, LEAF_D_MODEL = (args.dropout, args.mlp_hidden, args.leaf_d_model)
    USE_CALENDAR = not args.no_calendar
    LEAF_USE_SFM = not args.leaf_no_sfm
    LEAF_USE_CALENDAR = args.leaf_calendar
    LEAF_SFM_AXIS, LEAF_SFM_GATE = (args.leaf_sfm_axis, args.leaf_sfm_gate)
    NODE_PERM_SEED = args.node_perm_seed
    FUSION = args.fusion
    LOSS_SCALES, LOSS_LEVELS = (args.loss_scales, args.loss_levels)
    MLP_AUX_WEIGHT, LEAF_AUX_WEIGHT = (args.mlp_aux_weight, args.leaf_aux_weight)
    if args.device != 'auto':
        DEVICE = torch.device(args.device)
    raw_path_str = args.path[:-4] if args.path.endswith('.csv') else args.path
    p = Path(raw_path_str)
    if not p.is_absolute():
        if not p.parent.exists():
            p = ROOT / p
        p = p.resolve()
    DATA_PATH = str(p.with_name(f'{p.name}.csv').resolve())
    HIER_PATH = str(p.with_name(f'{p.name}_hier.csv').resolve())
    for f in (DATA_PATH, HIER_PATH):
        if not os.path.exists(f):
            raise FileNotFoundError(f'File not found: {f}')
    print(f'Loaded DATA_PATH: {DATA_PATH}')
    print(f'Loaded HIER_PATH: {HIER_PATH}')
    main()
if __name__ == '__main__':
    cli()

# %%
# Transformer-based baseline for MABe with sequence modeling.

import os
import json
import gc
import re
import itertools
import random
import time
import hashlib
from collections import defaultdict, deque
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

verbose = True

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using GPU? {torch.cuda.is_available()}")
if verbose and torch.cuda.is_available():
    props = torch.cuda.get_device_properties(device)
    print(f"  Device: {props.name} | total VRAM={props.total_memory / (1024 ** 3):.1f} GB")

def _env_int(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ('1','true','yes','y')


def _env_optional_int(name):
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None

def _slugify(text: str):
    cleaned = re.sub(r'[^0-9a-zA-Z]+', '-', str(text).strip()).strip('-').lower()
    if not cleaned:
        return 'default'
    max_len = 80
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{cleaned[:max_len - len(digest) - 1]}-{digest}"


def _ensure_dir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _load_threshold_file(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def _save_threshold_file(path: str, thresholds: dict):
    if not thresholds:
        return
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump({k: float(v) for k, v in thresholds.items()}, f)


USE_ADAPTIVE_THRESHOLDS = True        # False: 使用固定阈值
LOAD_THRESHOLDS = False               # True: 从磁盘加载阈值
LOAD_MODELS = False                   # True: 直接加载训练好的模型
CHECK_LOAD = True
THRESHOLD_DIR = "./threshold"
MODEL_DIR = "./models"
THRESHOLD_LOAD_DIR = THRESHOLD_DIR
MODEL_LOAD_DIR = MODEL_DIR


# GPU utilization controls (override defaults with environment variables like
# MABE_TRAIN_BATCH_SIZE, MABE_PRED_BATCH_SIZE, MABE_NUM_WORKERS, MABE_PIN_MEMORY,
# MABE_PERSISTENT_WORKERS, MABE_PREFETCH_FACTOR, and MABE_TORCH_COMPILE.)
def _auto_batch_size():
    if not torch.cuda.is_available():
        return 128
    try:
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    except RuntimeError:
        return 256
    if total_mem_gb >= 40:
        return 1024
    if total_mem_gb >= 24:
        return 768
    if total_mem_gb >= 16:
        return 512
    return 256


def _auto_num_workers():
    cpu_cnt = os.cpu_count() or 2
    return max(2, cpu_cnt // 2)


TRAIN_BATCH_SIZE = max(32, _env_int('MABE_TRAIN_BATCH_SIZE', _auto_batch_size()))
PRED_BATCH_SIZE = max(TRAIN_BATCH_SIZE, _env_int('MABE_PRED_BATCH_SIZE', TRAIN_BATCH_SIZE * 2))
NUM_WORKERS = max(0, _env_int('MABE_NUM_WORKERS', _auto_num_workers()))
PREFETCH_FACTOR = _env_int('MABE_PREFETCH_FACTOR', 4) if NUM_WORKERS > 0 else None
PIN_MEMORY = _env_bool('MABE_PIN_MEMORY', torch.cuda.is_available())
PERSISTENT_WORKERS = _env_bool('MABE_PERSISTENT_WORKERS', NUM_WORKERS > 0) and NUM_WORKERS > 0
USE_TORCH_COMPILE = _env_bool('MABE_TORCH_COMPILE', False) and hasattr(torch, "compile")

# Sequence / model hyper-parameters
SEQ_LEN = 128
TRAIN_STRIDE = 1
EPOCHS = 8
LR = 3e-4
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 2
FEEDFORWARD = 512
DROPOUT = 0.1
VAL_FRAC = 0.1
MIN_TRAIN_SAMPLES = 512
THRESHOLD_DEFAULT = 0.35
WINDOW_SMOOTHING = 7
TRAIN_N_SAMPLES = _env_optional_int('MABE_TRAIN_N_SAMPLES')

if verbose:
    print(
        f"  Train batch={TRAIN_BATCH_SIZE} | pred batch={PRED_BATCH_SIZE} | "
        f"workers={NUM_WORKERS} | pin_memory={PIN_MEMORY} | persistent_workers={PERSISTENT_WORKERS} | "
        f"torch.compile={USE_TORCH_COMPILE}"
    )

def _loader_kwargs(batch_size, sampler):
    kwargs = {
        'batch_size': batch_size,
        'sampler': sampler,
        'drop_last': False,
    }
    if NUM_WORKERS > 0:
        kwargs['num_workers'] = NUM_WORKERS
        kwargs['pin_memory'] = PIN_MEMORY
        kwargs['persistent_workers'] = PERSISTENT_WORKERS
        if PREFETCH_FACTOR is not None:
            kwargs['prefetch_factor'] = PREFETCH_FACTOR
    else:
        kwargs['num_workers'] = 0
        kwargs['pin_memory'] = False
    return kwargs

# %%
train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train = train.loc[~(train['lab_id'].astype(str).str.contains('MABe22', na=False) &
                    train['mouse1_condition'].astype(str).str.lower().eq('lights on'))].copy()
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
test['sleeping'] = (
    test['lab_id'].astype(str).str.contains('MABe22', na=False) &
    test['mouse1_condition'].astype(str).str.lower().eq('lights on')
)
test['n_mice'] = 4 - test[['mouse1_strain','mouse2_strain','mouse3_strain','mouse4_strain']].isna().sum(axis=1)

body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',                  
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

_sex_cols = [f'mouse{i}_sex' for i in range(1,5)]
_train_sex_lut = (train[['video_id'] + _sex_cols].drop_duplicates('video_id')
                  .set_index('video_id').to_dict('index'))
_test_sex_lut  = (test[['video_id']  + _sex_cols].drop_duplicates('video_id')
                  .set_index('video_id').to_dict('index'))
_FEATURE_TEMPLATES = {}

# %%
# ==================== DATA GENERATOR ====================

def generate_mouse_data(dataset, traintest, traintest_directory=None,
                        generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    def _to_num(x):
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r'(\d+)$', str(x))
        return int(m.group(1)) if m else None

    for _, row in dataset.iterrows():
        lab_id   = row.lab_id
        video_id = row.video_id
        fps      = float(row.frames_per_second)
        n_mice   = int(row.n_mice)
        arena_w  = float(row.get('arena_width_cm', np.nan))
        arena_h  = float(row.get('arena_height_cm', np.nan))
        sleeping = bool(getattr(row, 'sleeping', False))
        arena_shape = row.get('arena_shape', 'rectangular')

        if not isinstance(row.behaviors_labeled, str):
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id','bodypart'], index='video_frame', values=['x','y'])
        del vid
        pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
        pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False)

        avail = list(pvid.columns.get_level_values('mouse_id').unique())
        avail_set = set(avail) | set(map(str, avail)) | {f"mouse{_to_num(a)}" for a in avail if _to_num(a) is not None}

        def _resolve(agent_str):
            m = re.search(r'(\d+)$', str(agent_str))
            cand = [agent_str]
            if m:
                n = int(m.group(1))
                cand = [n, n-1, str(n), f"mouse{n}", agent_str]
            for c in cand:
                if c in avail_set:
                    if c in set(avail): return c
                    for a in avail:
                        if str(a) == str(c) or f"mouse{_to_num(a)}" == str(c):
                            return a
            return None

        vb = json.loads(row.behaviors_labeled)
        vb = sorted(list({b.replace("'", "") for b in vb}))
        vb = pd.DataFrame([b.split(',') for b in vb], columns=['agent','target','action'])
        vb['agent']  = vb['agent'].astype(str)
        vb['target'] = vb['target'].astype(str)
        vb['action'] = vb['action'].astype(str).str.lower()

        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        def _mk_meta(index, agent_id, target_id):
            m = pd.DataFrame({
                'lab_id':        lab_id,
                'video_id':      video_id,
                'agent_id':      agent_id,
                'target_id':     target_id,
                'video_frame':   index.astype('int32', copy=False),
                'frames_per_second': np.float32(fps),
                'sleeping':      sleeping,
                'arena_shape':   arena_shape,
                'arena_width_cm': np.float32(arena_w),
                'arena_height_cm': np.float32(arena_h),
                'n_mice':        np.int8(n_mice),
            })
            for c in ('lab_id','video_id','agent_id','target_id','arena_shape'):
                m[c] = m[c].astype('category')
            return m

        if generate_single:
            vb_single = vb.query("target == 'self'")
            for agent_str in pd.unique(vb_single['agent']):
                col_lab = _resolve(agent_str)
                if col_lab is None:
                    continue
                actions = sorted(vb_single.loc[vb_single['agent'].eq(agent_str), 'action'].unique().tolist())
                if not actions:
                    continue

                single = pvid.loc[:, col_lab]
                meta_df = _mk_meta(single.index, agent_str, 'self')

                if traintest == 'train':
                    a_num = _to_num(col_lab)
                    y = pd.DataFrame(False, index=single.index.astype('int32', copy=False), columns=actions)
                    a_sub = annot.query("(agent_id == @a_num) & (target_id == @a_num)")
                    for i in range(len(a_sub)):
                        ar = a_sub.iloc[i]
                        a = str(ar.action).lower()
                        if a in y.columns:
                            y.loc[int(ar['start_frame']):int(ar['stop_frame']), a] = True
                    yield 'single', single, meta_df, y
                else:
                    yield 'single', single, meta_df, actions

        if generate_pair:
            vb_pair = vb.query("target != 'self'")
            if len(vb_pair) > 0:
                allowed_pairs = set(map(tuple, vb_pair[['agent','target']].itertuples(index=False, name=None)))

                for agent_num, target_num in itertools.permutations(
                        np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{_to_num(agent_num)}"
                    target_str = f"mouse{_to_num(target_num)}"
                    if (agent_str, target_str) not in allowed_pairs:
                        continue

                    a_col = _resolve(agent_str)
                    b_col = _resolve(target_str)
                    if a_col is None or b_col is None:
                        continue

                    actions = sorted(
                        vb_pair.query("(agent == @agent_str) & (target == @target_str)")['action'].unique().tolist()
                    )
                    if not actions:
                        continue

                    pair_xy = pd.concat([pvid[a_col], pvid[b_col]], axis=1, keys=['A','B'])
                    meta_df = _mk_meta(pair_xy.index, agent_str, target_str)

                    if traintest == 'train':
                        a_num = _to_num(a_col); b_num = _to_num(b_col)
                        y = pd.DataFrame(False, index=pair_xy.index.astype('int32', copy=False), columns=actions)
                        a_sub = annot.query("(agent_id == @a_num) & (target_id == @b_num)")
                        for i in range(len(a_sub)):
                            ar = a_sub.iloc[i]
                            a = str(ar.action).lower()
                            if a in y.columns:
                                y.loc[int(ar['start_frame']):int(ar['stop_frame']), a] = True
                        yield 'pair', pair_xy, meta_df, y
                    else:
                        yield 'pair', pair_xy, meta_df, actions

# %%
def predict_multiclass_adaptive(pred, meta, action_thresholds=defaultdict(lambda: THRESHOLD_DEFAULT)):
    if 'video_frame' not in meta.columns:
        if meta.index.name == 'video_frame':
            meta = meta.reset_index().rename(columns={'index': 'video_frame'})
        elif 'index' in meta.columns:
            meta = meta.rename(columns={'index': 'video_frame'})
        else:
            meta = meta.copy()
            meta['video_frame'] = meta.index.to_numpy()

    pred_smoothed = pred.rolling(window=WINDOW_SMOOTHING, min_periods=1, center=True).mean()
    thresholds = np.array([action_thresholds.get(action, THRESHOLD_DEFAULT) for action in pred_smoothed.columns], dtype=np.float32)
    vals = pred_smoothed.values
    valid = vals >= thresholds
    masked = np.where(valid, vals, -np.inf)
    ama = masked.argmax(axis=1)
    all_invalid = ~valid.any(axis=1)
    ama[all_invalid] = -1
    ama = pd.Series(ama, index=meta['video_frame'].values)

    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    if len(mask) == 0:
        return pd.DataFrame(columns=['video_id','agent_id','target_id','action','start_frame','stop_frame'])
    mask[-1] = False

    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })

    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values if len(mask) > 1 else []
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values if len(mask) > 1 else []
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values if len(mask) > 1 else []

    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame

    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    return submission_part

# %%
def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]

    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop:
                mask[i] = False
            else:
                last_stop = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list) if group_list else submission

    s_list = []
    for _, row in dataset.iterrows():
        lab_id = row['lab_id']
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = eval(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])

    submission = submission.reset_index(drop=True)
    return submission

# %%
def _ensure_template(key, df):
    if key not in _FEATURE_TEMPLATES:
        _FEATURE_TEMPLATES[key] = df.columns.tolist()
    return df.reindex(columns=_FEATURE_TEMPLATES[key], fill_value=0.0)


def _extract_single_features(single_mouse: pd.DataFrame, body_parts: List[str]):
    idx = single_mouse.index
    available = set(single_mouse.columns.get_level_values(0))
    feats = {}
    for part in body_parts:
        if part in available:
            coords = single_mouse[part]
            feats[f'{part}_x'] = coords['x']
            feats[f'{part}_y'] = coords['y']
        else:
            feats[f'{part}_x'] = pd.Series(np.nan, index=idx)
            feats[f'{part}_y'] = pd.Series(np.nan, index=idx)
    for p1, p2 in itertools.combinations(body_parts, 2):
        col = f'd_{p1}_{p2}'
        if p1 in available and p2 in available:
            dx = single_mouse[p1]['x'] - single_mouse[p2]['x']
            dy = single_mouse[p1]['y'] - single_mouse[p2]['y']
            feats[col] = np.sqrt(dx**2 + dy**2)
        else:
            feats[col] = pd.Series(np.nan, index=idx)
    df = pd.DataFrame(feats)
    df = df.astype('float32')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)
    return df


def _extract_pair_features(mouse_pair: pd.DataFrame, body_parts: List[str]):
    idx = mouse_pair.index
    avail_A = set(mouse_pair['A'].columns.get_level_values(0))
    avail_B = set(mouse_pair['B'].columns.get_level_values(0))
    feats = {}
    for part in body_parts:
        if part in avail_A:
            feats[f'A_{part}_x'] = mouse_pair['A'][part]['x']
            feats[f'A_{part}_y'] = mouse_pair['A'][part]['y']
        else:
            feats[f'A_{part}_x'] = pd.Series(np.nan, index=idx)
            feats[f'A_{part}_y'] = pd.Series(np.nan, index=idx)
        if part in avail_B:
            feats[f'B_{part}_x'] = mouse_pair['B'][part]['x']
            feats[f'B_{part}_y'] = mouse_pair['B'][part]['y']
        else:
            feats[f'B_{part}_x'] = pd.Series(np.nan, index=idx)
            feats[f'B_{part}_y'] = pd.Series(np.nan, index=idx)
        if part in avail_A and part in avail_B:
            dx = mouse_pair['A'][part]['x'] - mouse_pair['B'][part]['x']
            dy = mouse_pair['A'][part]['y'] - mouse_pair['B'][part]['y']
            feats[f'diff_{part}_x'] = dx
            feats[f'diff_{part}_y'] = dy
        else:
            feats[f'diff_{part}_x'] = pd.Series(np.nan, index=idx)
            feats[f'diff_{part}_y'] = pd.Series(np.nan, index=idx)
    key_pairs = [('nose','nose'), ('body_center','body_center'), ('tail_base','tail_base')]
    for pA, pB in key_pairs:
        col = f'd_{pA}_{pB}'
        if pA in avail_A and pB in avail_B:
            dx = mouse_pair['A'][pA]['x'] - mouse_pair['B'][pB]['x']
            dy = mouse_pair['A'][pA]['y'] - mouse_pair['B'][pB]['y']
            feats[col] = np.sqrt(dx**2 + dy**2)
        else:
            feats[col] = pd.Series(np.nan, index=idx)
    df = pd.DataFrame(feats)
    df = df.astype('float32')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)
    return df


def _compute_scaler(X_df: pd.DataFrame):
    mean = X_df.mean().astype('float32')
    std = X_df.std().replace(0, 1.0).astype('float32')
    return mean, std


def _normalize_features(X_df: pd.DataFrame, columns: List[str], mean: pd.Series, std: pd.Series):
    aligned = X_df.reindex(columns=columns, fill_value=0.0)
    normed = (aligned - mean) / std
    return normed.fillna(0.0)


def _build_groups(X_df: pd.DataFrame, y_df: pd.DataFrame, meta_df: pd.DataFrame):
    groups = []
    if len(X_df) == 0:
        return groups
    X_df = X_df.reset_index(drop=True)
    y_df = y_df.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)
    group_cols = ['video_id','agent_id','target_id']
    grouped = meta_df.groupby(group_cols, sort=False, observed=False)
    for _, idx in grouped.groups.items():
        idx = np.array(idx)
        frames = meta_df.loc[idx, 'video_frame'].values
        order = np.argsort(frames, kind='stable')
        idx = idx[order]
        X_arr = X_df.iloc[idx].to_numpy(np.float32, copy=True)
        y_arr = y_df.iloc[idx].to_numpy(np.float32, copy=True)
        meta_part = meta_df.iloc[idx].reset_index(drop=True)
        if len(X_arr) == 0:
            continue
        groups.append({'X': X_arr, 'y': y_arr, 'meta': meta_part})
    return groups


def _select_stride(groups: List[dict], base_stride: int, target_samples: int | None):
    if target_samples is None or target_samples <= 0:
        return base_stride
    total_frames = sum(len(g['X']) for g in groups)
    if total_frames == 0:
        return base_stride
    approx_stride = max(1, int(np.ceil(total_frames / target_samples)))
    return approx_stride


class SlidingWindowDataset(Dataset):
    def __init__(self, groups: List[dict], seq_len: int, stride: int):
        self.groups = groups
        self.seq_len = seq_len
        self.index = []
        self.group_video_ids = []
        for gi, group in enumerate(groups):
            n = len(group['X'])
            video_id = str(group['meta']['video_id'].iloc[0])
            for t in range(0, n, stride):
                self.index.append((gi, t))
                self.group_video_ids.append(video_id)
        self.length = len(self.index)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gi, t = self.index[idx]
        group = self.groups[gi]
        Xg = group['X']
        yg = group['y']
        start = max(0, t - self.seq_len + 1)
        window = Xg[start:t+1]
        if len(window) < self.seq_len:
            pad = np.repeat(window[:1], self.seq_len - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        return torch.from_numpy(window), torch.from_numpy(yg[t])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_outputs: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL, dropout=DROPOUT, max_len=SEQ_LEN + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FEEDFORWARD,
            batch_first=True, dropout=DROPOUT, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, num_outputs)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        last = x[:, -1, :]
        return self.head(last)


def _micro_f1(tp: float, fp: float, fn: float) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def _grid_search_threshold(pred: np.ndarray, target: np.ndarray) -> float:
    uniq = np.unique(target)
    if uniq.size < 2:
        return THRESHOLD_DEFAULT
    best_thr = THRESHOLD_DEFAULT
    best_f1 = -1.0
    for thr in np.linspace(0.0, 1.0, 101):
        preds = pred >= thr
        tp = np.logical_and(preds, target).sum()
        fp = np.logical_and(preds, ~target).sum()
        fn = np.logical_and(~preds, target).sum()
        denom = (2 * tp) + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _tune_action_thresholds(prob: np.ndarray, target: np.ndarray, actions: List[str]) -> dict:
    tuned = {}
    bool_target = target >= 0.5
    for idx, action in enumerate(actions):
        thr = _grid_search_threshold(prob[:, idx], bool_target[:, idx])
        tuned[action] = thr
    return tuned


def _train_transformer(groups: List[dict], input_dim: int, n_outputs: int, action_names: List[str], tune_thresholds: bool):
    stride = _select_stride(groups, TRAIN_STRIDE, TRAIN_N_SAMPLES)
    dataset = SlidingWindowDataset(groups, seq_len=SEQ_LEN, stride=stride)
    if len(dataset) < MIN_TRAIN_SAMPLES:
        if verbose:
            print(f"  Skipping transformer training (only {len(dataset)} samples)")
        return None, {}
    if verbose and TRAIN_N_SAMPLES:
        total_frames = sum(len(g['X']) for g in groups)
        print(
            f"  Adaptive stride={stride} | target_samples={TRAIN_N_SAMPLES} | "
            f"frames={total_frames} | actual_samples={len(dataset)}"
        )
    video_ids = np.array(dataset.group_video_ids)
    unique_vids = np.unique(video_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_vids)
    target_val_vids = max(1, int(len(unique_vids) * VAL_FRAC))
    if len(unique_vids) > 1:
        val_vids = set(unique_vids[:target_val_vids])
    else:
        val_vids = set()
    train_idx = np.nonzero(~np.isin(video_ids, list(val_vids)))[0]
    val_idx = np.nonzero(np.isin(video_ids, list(val_vids)))[0]
    if len(train_idx) == 0:
        train_idx, val_idx = val_idx, np.array([], dtype=int)
    train_loader = DataLoader(dataset, **_loader_kwargs(TRAIN_BATCH_SIZE, SubsetRandomSampler(train_idx)))
    val_loader = (DataLoader(dataset, **_loader_kwargs(TRAIN_BATCH_SIZE, SubsetRandomSampler(val_idx)))
                  if len(val_idx) > 0 else None)

    model = TransformerClassifier(input_dim, n_outputs).to(device)
    if USE_TORCH_COMPILE:
        try:
            if verbose:
                print("  Compiling transformer with torch.compile for higher throughput")
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - compile path optional
            if verbose:
                print(f"  torch.compile unavailable ({exc}); continuing without it")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float('inf')
    tuned_thresholds = {}
    epoch_times = deque(maxlen=5)
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        train_tp = train_fp = train_fn = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # accumulate micro-F1 counts on-the-fly
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                preds = prob >= THRESHOLD_DEFAULT
                target = yb >= 0.5
                train_tp += torch.logical_and(preds, target).sum().item()
                train_fp += torch.logical_and(preds, ~target).sum().item()
                train_fn += torch.logical_and(~preds, target).sum().item()
            total_loss += loss.item() * len(xb)
            total_count += len(xb)
        avg_loss = total_loss / max(1, total_count)
        train_f1 = _micro_f1(train_tp, train_fp, train_fn)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            val_tp = val_fp = val_fn = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    prob = torch.sigmoid(logits)
                    preds = prob >= THRESHOLD_DEFAULT
                    target = yb >= 0.5
                    val_tp += torch.logical_and(preds, target).sum().item()
                    val_fp += torch.logical_and(preds, ~target).sum().item()
                    val_fn += torch.logical_and(~preds, target).sum().item()
                    val_loss += loss.item() * len(xb)
                    val_count += len(xb)
            val_loss = val_loss / max(1, val_count)
            val_f1 = _micro_f1(val_tp, val_fp, val_fn)
        else:
            val_loss = avg_loss
            val_f1 = train_f1

        if verbose:
            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)
            remaining = (sum(epoch_times) / max(1, len(epoch_times))) * (EPOCHS - epoch - 1)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
            print(
                f"    epoch {epoch+1}/{EPOCHS} | "
                f"train_loss={avg_loss:.4f} | train_f1={train_f1:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | ETA {eta_str}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if tune_thresholds and val_loader is not None:
        val_probs = []
        val_targets = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(prob)
                val_targets.append(yb.cpu().numpy())
        if val_probs:
            val_probs = np.concatenate(val_probs, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            tuned_thresholds = _tune_action_thresholds(val_probs, val_targets, action_names)
    return model, dict(tuned_thresholds)


def _align_feature_meta(X_df: pd.DataFrame, meta_df: pd.DataFrame):
    meta_sorted = meta_df.sort_values('video_frame').reset_index(drop=True)
    index_order = meta_sorted['video_frame'].values
    X_sorted = X_df.reindex(index_order)
    X_sorted = X_sorted.ffill().bfill().fillna(0.0)
    return X_sorted, meta_sorted


def _predict_video(model, X_df, meta_df, feature_cols, mean, std, actions):
    if len(X_df) == 0:
        empty_meta = meta_df.iloc[0:0].copy()
        return pd.DataFrame(columns=actions), empty_meta
    X_aligned, meta_aligned = _align_feature_meta(X_df, meta_df)
    normed = _normalize_features(X_aligned, feature_cols, mean, std)
    arr = normed.to_numpy(np.float32, copy=True)
    frames = meta_aligned['video_frame'].astype(int).tolist()
    preds = []
    frame_accum = []
    batch_windows = []
    batch_frames = []
    for i in range(len(arr)):
        start = max(0, i - SEQ_LEN + 1)
        window = arr[start:i+1]
        if len(window) < SEQ_LEN:
            pad = np.repeat(window[:1], SEQ_LEN - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        batch_windows.append(window)
        batch_frames.append(frames[i])
        if len(batch_windows) >= PRED_BATCH_SIZE or i == len(arr) - 1:
            xb = torch.from_numpy(np.stack(batch_windows)).to(device)
            with torch.no_grad():
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
            preds.append(prob)
            frame_accum.extend(batch_frames)
            batch_windows = []
            batch_frames = []
    if not preds:
        return pd.DataFrame(columns=actions)
    probs = np.concatenate(preds, axis=0)
    pred_df = pd.DataFrame(probs, index=frame_accum, columns=actions)
    pred_df = pred_df.sort_index()
    meta_sorted = meta_aligned.set_index('video_frame').loc[pred_df.index].reset_index()
    return pred_df, meta_sorted


def run_transformer_pipeline(body_parts_tracked_str, switch, X_df, label_df, meta_df, test_subset, body_parts, submission_list):
    if len(X_df) == 0:
        return
    label_df = label_df.fillna(0.0)
    action_counts = label_df.sum(axis=0, skipna=True)
    actions = [a for a in label_df.columns if action_counts.get(a, 0) > 0]
    if not actions:
        if verbose:
            print(f"  Skipping {switch} | {body_parts_tracked_str} (no positive actions)")
        return
    label_df = label_df[actions].astype('float32')
    template_key = (switch, body_parts_tracked_str)
    X_df = _ensure_template(template_key, X_df).astype('float32')
    mean, std = _compute_scaler(X_df)
    X_norm = _normalize_features(X_df, X_df.columns.tolist(), mean, std)
    groups = _build_groups(X_norm, label_df, meta_df)
    if not groups:
        if verbose:
            print(f"  Skipping {switch} | {body_parts_tracked_str} (no training groups)")
        return
    slug = _slugify(body_parts_tracked_str)
    model_fname = f"{slug}_{switch}_transformer.pt"
    thr_fname = f"{slug}_{switch}_thresholds.json"
    model = None
    tuned_thresholds = {}
    model_loaded = False
    if LOAD_MODELS:
        model_load_path = os.path.join(MODEL_LOAD_DIR, model_fname)
        if os.path.exists(model_load_path):
            try:
                model = TransformerClassifier(X_norm.shape[1], len(actions)).to(device)
                state = torch.load(model_load_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                model_loaded = True
                if verbose:
                    print(f"  Loaded transformer weights from {model_load_path}")
            except Exception as exc:
                model = None
                if verbose:
                    print(f"  Failed to load model ({exc}); retraining.")
    if model is None:
        if verbose:
            print(f"  Training transformer on {len(groups)} groups ({len(X_norm)} frames)")
        model, tuned_thresholds = _train_transformer(
            groups,
            input_dim=X_norm.shape[1],
            n_outputs=len(actions),
            action_names=actions,
            tune_thresholds=USE_ADAPTIVE_THRESHOLDS
        )
        if model is None:
            return
        model.eval()
        if MODEL_DIR:
            model_save_path = os.path.join(MODEL_DIR, model_fname)
            try:
                _ensure_dir(MODEL_DIR)
                torch.save(model.state_dict(), model_save_path)
                if verbose:
                    print(f"  Saved transformer -> {model_save_path}")
            except OSError as exc:
                if verbose:
                    print(f"  Failed to save model ({exc})")
    action_thresholds = defaultdict(lambda: THRESHOLD_DEFAULT)
    if USE_ADAPTIVE_THRESHOLDS:
        thresholds_data = {}
        thr_load_path = os.path.join(THRESHOLD_LOAD_DIR, thr_fname)
        if LOAD_THRESHOLDS and os.path.exists(thr_load_path):
            try:
                thresholds_data.update(_load_threshold_file(thr_load_path))
                if verbose:
                    print(f"  Loaded thresholds from {thr_load_path}")
            except Exception as exc:
                if verbose:
                    print(f"  Failed to load thresholds ({exc}); falling back.")
        if tuned_thresholds:
            thresholds_data.update(tuned_thresholds)
            thr_save_path = os.path.join(THRESHOLD_DIR, thr_fname)
            try:
                _save_threshold_file(thr_save_path, tuned_thresholds)
                if verbose:
                    print(f"  Saved thresholds -> {thr_save_path}")
            except OSError as exc:
                if verbose:
                    print(f"  Failed to save thresholds ({exc})")
        action_thresholds.update(thresholds_data)
        if CHECK_LOAD:
            missing = [a for a in actions if a not in action_thresholds]
            if missing and verbose:
                print(f"  Warning: missing thresholds for {missing}, using default {THRESHOLD_DEFAULT:.2f}")
    else:
        if verbose:
            print("  Using constant thresholds (adaptive disabled)")

    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch == 'single'),
        generate_pair=(switch == 'pair')
    )
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch
        if switch == 'single':
            feat = _extract_single_features(data_te, body_parts)
        else:
            feat = _extract_pair_features(data_te, body_parts)
        feat = _ensure_template(template_key, feat)
        pred_df, meta_sorted = _predict_video(model, feat, meta_te, X_df.columns.tolist(), mean, std, actions)
        if pred_df.empty:
            continue
        actions_available = [a for a in actions_te if a in actions]
        if not actions_available:
            continue
        submission_list.append(predict_multiclass_adaptive(pred_df[actions_available], meta_sorted, action_thresholds))
    torch.cuda.empty_cache()

# %%
submission_list = []

section_times = deque(maxlen=3)
for section, body_parts_tracked_str in enumerate(body_parts_tracked_list):
    iter_start = time.time()
    body_parts_tracked = json.loads(body_parts_tracked_str)
    print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]

    single_feats_parts, single_label_list, single_meta_list = [], [], []
    pair_feats_parts, pair_label_list, pair_meta_list = [], [], []

    for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
        if switch == 'single':
            feat = _extract_single_features(data, body_parts_tracked)
            key = ('single', body_parts_tracked_str)
            feat = _ensure_template(key, feat)
            meta_sorted = meta.sort_values('video_frame').reset_index(drop=True)
            feat = feat.reindex(meta_sorted['video_frame'].values).reset_index(drop=True)
            label = label.reindex(meta_sorted['video_frame'].values).reset_index(drop=True)
            single_feats_parts.append(feat)
            single_label_list.append(label)
            single_meta_list.append(meta_sorted)
        else:
            feat = _extract_pair_features(data, body_parts_tracked)
            key = ('pair', body_parts_tracked_str)
            feat = _ensure_template(key, feat)
            meta_sorted = meta.sort_values('video_frame').reset_index(drop=True)
            feat = feat.reindex(meta_sorted['video_frame'].values).reset_index(drop=True)
            label = label.reindex(meta_sorted['video_frame'].values).reset_index(drop=True)
            pair_feats_parts.append(feat)
            pair_label_list.append(label)
            pair_meta_list.append(meta_sorted)

    if single_feats_parts:
        X_single = pd.concat(single_feats_parts, axis=0, ignore_index=True)
        y_single = pd.concat(single_label_list, axis=0, ignore_index=True)
        meta_single = pd.concat(single_meta_list, axis=0, ignore_index=True)
        print(f"  Single frames: {len(X_single)}")
        run_transformer_pipeline(body_parts_tracked_str, 'single', X_single, y_single, meta_single, test_subset, body_parts_tracked, submission_list)
        del X_single, y_single, meta_single
        gc.collect()

    if pair_feats_parts:
        X_pair = pd.concat(pair_feats_parts, axis=0, ignore_index=True)
        y_pair = pd.concat(pair_label_list, axis=0, ignore_index=True)
        meta_pair = pd.concat(pair_meta_list, axis=0, ignore_index=True)
        print(f"  Pair frames: {len(X_pair)}")
        run_transformer_pipeline(body_parts_tracked_str, 'pair', X_pair, y_pair, meta_pair, test_subset, body_parts_tracked, submission_list)
        del X_pair, y_pair, meta_pair
        gc.collect()

    iter_elapsed = time.time() - iter_start
    section_times.append(iter_elapsed)
    remaining_sections = len(body_parts_tracked_list) - section - 1
    if remaining_sections > 0 and section_times:
        avg_section = sum(section_times) / len(section_times)
        eta_sections = avg_section * remaining_sections
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sections))
        print(f"  Section ETA: {eta_str}")
    else:
        print()

if len(submission_list) > 0:
    submission = pd.concat(submission_list, ignore_index=True)
else:
    submission = pd.DataFrame({
        'video_id': [438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [278],
        'stop_frame': [500]
    })

submission_robust = robustify(submission, test, 'test')
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
print(f"\nSubmission created: {len(submission_robust)} predictions")

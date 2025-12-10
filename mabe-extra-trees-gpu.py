# %%
# Update of AmbrosMs' great notebook
# Removes the constant FPS assumption; handles variable frame timing
# Added full GPU suppoprt and extra trees/feats

verbose = True

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import warnings
import json
import os, random
import gc, re, math
import hashlib
import joblib
from collections import defaultdict
import polars as pl
from scipy import signal, stats
from typing import Dict, Optional, Tuple
from time import perf_counter 
import optuna
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
USE_GPU = ("KAGGLE_KERNEL_RUN_TYPE" in __import__("os").environ) and (__import__("shutil").which("nvidia-smi") is not None)
print(f'Using GPU? {USE_GPU}')

from xgboost import XGBClassifier

SEED = 1234
N_CUT = 1_000_000_000
LEAST_POS_PERC = 0.0
_safe_token = re.compile(r'[^A-Za-z0-9]+')
# ---- runtime switches ----
ONLY_TUNE_THRESHOLDS = False     # True: only search thresholds and save; skip training/inference/submission
USE_ADAPTIVE_THRESHOLDS = True   # False: use constant 0.27 for all actions, skip tuning/loading
LOAD_THRESHOLDS = False          # True: load thresholds from THRESHOLD_DIR instead of tuning
LOAD_MODELS = False              # True: load models from MODEL_DIR instead of training
CHECK_LOAD = True
THRESHOLD_DIR = "./threshold"            # save thresholds here
MODEL_DIR = "./models"                  # save models here
THRESHOLD_LOAD_DIR = THRESHOLD_DIR      # load thresholds from here
MODEL_LOAD_DIR = MODEL_DIR              # load models from here

# %%
# --- SEED EVERYTHING -----
os.environ["PYTHONHASHSEED"] = str(SEED)      # has to be set very early

rnd = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)

def _make_xgb(**kw):
    kw.setdefault("random_state", SEED)
    kw.setdefault("tree_method", "gpu_hist" if USE_GPU else "hist")
    # kw.setdefault("deterministic_histogram", True)
    return XGBClassifier(**kw)

def _slugify(text: str) -> str:
    """Make a filename-safe slug from the body_parts_tracked string (truncate + hash to avoid long paths)."""
    base = _safe_token.sub('-', text).strip('-') or "default"
    max_len = 80
    if len(base) <= max_len:
        return base
    digest = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{base[:max_len - len(digest) - 1]}-{digest}"


# %%
# ================= StratifiedSubsetClassifier =================
class StratifiedSubsetClassifierWEval(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 estimator,
                 n_samples=None,
                 random_state: int = 42,
                 valid_size: float = 0.10,
                 val_cap_ratio: float = 0.25,
                 es_rounds: "int|str" = "auto",
                 es_metric: str = "auto"):
        self.estimator = estimator
        self.n_samples = (int(n_samples) if (n_samples is not None) else None)
        self.random_state = random_state
        self.valid_size = float(valid_size)
        self.val_cap_ratio = float(val_cap_ratio)
        self.es_rounds = es_rounds
        self.es_metric = es_metric
 
    # -------------------------- API --------------------------
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        n_total = len(y); assert n_total == len(X)

        tr_idx, va_idx = self._compute_train_val_indices(y, n_total)
        Xtr = X.iloc[tr_idx]; ytr = y[tr_idx]

        Xtr = Xtr.to_numpy(np.float32, copy=False)

        Xva = yva = None
        if va_idx is not None and len(va_idx) > 0:
            Xva = X.iloc[va_idx].to_numpy(np.float32, copy=False); yva = y[va_idx]

        # Compute pos_rate on VALIDATION (what ES monitors)
        pos_rate = None
        if yva is not None and len(yva) > 0:
            pos_rate = float(np.mean(yva == 1))

        # Decide metric & patience
        metric = self._choose_metric(pos_rate)
        patience = self._choose_patience(pos_rate)

        # Apply imbalance knobs per library
        if self._is_xgb(self.estimator):
            # scale_pos_weight = n_neg / n_pos on TRAIN
            n_pos = max(1, int((ytr == 1).sum()))
            n_neg = max(1, len(ytr) - n_pos)
            self.estimator.set_params(scale_pos_weight=(n_neg / n_pos))
            self.estimator.set_params(eval_metric=metric)

        # Fit with ES if we have any validation (single-class OK with Logloss)
        has_valid = (Xva is not None and len(yva) > 0)
        if has_valid and self._is_xgb(self.estimator):
            import xgboost as xgb
            self.estimator.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                verbose=False,
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=int(patience),
                    metric_name=metric,
                    data_name="validation_0",
                    save_best=True
                )]
            )
        else:
            # Fall back: train on train split without ES
            self.estimator.fit(Xtr, ytr)

        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        self._tr_idx_ = tr_idx; self._va_idx_ = va_idx; self._pos_rate_ = pos_rate
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(X)

    # -------------------------- helpers --------------------------
    def _compute_train_val_indices(self, y: np.ndarray, n_total: int):
        rng = np.random.default_rng(self.random_state)
        n_classes = np.unique(y).size

        def full_data_split():
            if self.valid_size <= 0 or n_classes < 2:
                idx = rng.permutation(n_total); return idx, None
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.valid_size, random_state=self.random_state)
            tr, va = next(sss.split(np.zeros(n_total, dtype=np.int8), y))
            return tr, va

        if self.n_samples is None or self.n_samples >= n_total:
            return full_data_split()

        # Use n_samples for train; build val from remainder (capped)
        sss_tr = StratifiedShuffleSplit(n_splits=1, train_size=self.n_samples, random_state=self.random_state)
        tr_idx, rest_idx = next(sss_tr.split(np.zeros(n_total, dtype=np.int8), y))
        remaining = len(rest_idx)

        min_val_needed = int(np.ceil(self.n_samples * max(self.valid_size, 0.0)))
        val_cap = max(min_val_needed, int(round(self.val_cap_ratio * self.n_samples)))
        want_val = min(remaining, val_cap)

        y_rest = y[rest_idx]
        if remaining < min_val_needed or np.unique(y_rest).size < 2 or self.valid_size <= 0:
            return full_data_split()

        sss_val = StratifiedShuffleSplit(n_splits=1, train_size=want_val, random_state=self.random_state)
        try:
            va_sel, _ = next(sss_val.split(np.zeros(remaining, dtype=np.int8), y_rest))
        except ValueError:
            return full_data_split()

        va_idx = rest_idx[va_sel]
        return tr_idx, va_idx

    def _choose_metric(self, pos_rate=0.01) -> str:
        if self.es_metric != "auto":
            return self.es_metric
        if pos_rate is None or pos_rate == 0.0 or pos_rate == 1.0:
            return "logloss" if self._is_xgb(self.estimator) else "Logloss"
        return "aucpr" if self._is_xgb(self.estimator) else "PRAUC:type=Classic"

    def _choose_patience(self, pos_rate: Optional[float]) -> int:
        if isinstance(self.es_rounds, int):
            return self.es_rounds
        try:
            n_estimators = (int(self.estimator.get_params().get("n_estimators", 200))
                            if self._is_xgb(self.estimator)
                            else int(self.estimator.get_params().get("iterations", 500)))
        except Exception:
            n_estimators = 200
        base = max(30, int(round(0.20 * (n_estimators or 200))))
        if pos_rate is None:
            return base
        if pos_rate < 0.005:   # <0.5%
            return int(round(base * 1.75))
        if pos_rate < 0.02:    # <2%
            return int(round(base * 1.40))
        return base

    @staticmethod
    def _is_xgb(est):
        name = est.__class__.__name__.lower(); mod = getattr(est, "__module__", "")
        return "xgb" in name or "xgboost" in mod or hasattr(est, "get_xgb_params")

    @staticmethod
    def _is_catboost(est):
        name = est.__class__.__name__.lower(); mod = getattr(est, "__module__", "")
        return "catboost" in name or "catboost" in mod or hasattr(est, "get_all_params")


class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_samples, random_state=SEED):
        self.estimator = estimator
        self.n_samples = n_samples and int(n_samples)
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        n_total = len(y)

        if self.n_samples is None or self.n_samples >= n_total:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(n_total)
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=self.n_samples, random_state=self.random_state
            )
            idx, _ = next(sss.split(np.zeros(n_total, dtype=np.int8), y))

        Xn = X.iloc[idx]
        Xn = Xn.to_numpy(np.float32, copy=False)
        yn = y[idx]

        self.estimator.fit(Xn, yn)
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)


# %%
# ==================== SCORING FUNCTIONS ====================

class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

# %%
# ==================== DATA LOADING ====================

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')

# drop likely-sleeping MABe22 clips: condition == "lights on"
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

        # ---- tracking ----
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id','bodypart'], index='video_frame', values=['x','y'])
        del vid
        pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
        pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False)

        # available mouse_id labels in tracking (could be ints or strings)
        avail = list(pvid.columns.get_level_values('mouse_id').unique())
        avail_set = set(avail) | set(map(str, avail)) | {f"mouse{_to_num(a)}" for a in avail if _to_num(a) is not None}

        def _resolve(agent_str):
            """Return the matching mouse_id label present in pvid (int or str), or None."""
            m = re.search(r'(\d+)$', str(agent_str))
            cand = [agent_str]
            if m:
                n = int(m.group(1))
                cand = [n, n-1, str(n), f"mouse{n}", agent_str]  # try 1-based, 0-based, str, canonical
            for c in cand:
                if c in avail_set:  # compare within unified set
                    # return the exact label used in columns
                    if c in set(avail): return c
                    # map back to the exact label that exists (int preferred)
                    for a in avail:
                        if str(a) == str(c) or f"mouse{_to_num(a)}" == str(c):
                            return a
            return None

        # ---- behaviors ----
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
            # Align index to frame numbers so sampling with .loc on frame ids works.
            # m = m.set_index('video_frame')
            # m['video_frame'] = m.index
            return m

        # ---------- SINGLE ----------
        if generate_single:
            vb_single = vb.query("target == 'self'")
            for agent_str in pd.unique(vb_single['agent']):
                col_lab = _resolve(agent_str)
                if col_lab is None:
                    # if verbose: print(f"[skip single] {video_id} missing {agent_str} in tracking (avail={sorted(avail)})")
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

        # ---------- PAIR (ONLY LABELED PAIRS) ----------
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
                        # if verbose: print(f"[skip pair] {video_id} missing {agent_str}->{target_str}")
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
# ==================== ADAPTIVE THRESHOLDING ====================

def predict_multiclass_adaptive(pred, meta, action_thresholds=defaultdict(lambda: 0.27)):
    """Adaptive thresholding per action + temporal smoothing"""
    # Apply temporal smoothing
    pred_smoothed = pred.rolling(window=7, min_periods=1, center=True).mean()
    
    # Apply per-action thresholds first, then pick max among the valid ones
    thresholds = np.array([action_thresholds.get(action, 0.27) for action in pred_smoothed.columns], dtype=np.float32)
    vals = pred_smoothed.values
    valid = vals >= thresholds  # broadcast per action
    masked = np.where(valid, vals, -np.inf)
    ama = masked.argmax(axis=1)
    all_invalid = ~valid.any(axis=1)
    ama[all_invalid] = -1
    ama = pd.Series(ama, index=meta.video_frame)
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    
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
    
    # Filter out very short events (likely noise)
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: print(f'  actions found: {len(submission_part)}')
    return submission_part

# %%
# ==================== ADVANCED FEATURE ENGINEERING (FPS-AWARE) ====================

def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts (keeps at least 1 frame when |n|>=1)."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def _speed(cx: pd.Series, cy: pd.Series, fps: float) -> pd.Series:
    return np.hypot(cx.diff(), cy.diff()).fillna(0.0) * float(fps)

def _roll_future_mean(s: pd.Series, w: int, min_p: int = 1) -> pd.Series:
    # mean over [t, t+w-1]
    return s.iloc[::-1].rolling(w, min_periods=min_p).mean().iloc[::-1]

def _roll_future_var(s: pd.Series, w: int, min_p: int = 2) -> pd.Series:
    # var over [t, t+w-1]
    return s.iloc[::-1].rolling(w, min_periods=min_p).var().iloc[::-1]


def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)."""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)  # invariant to time scaling

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s; windows scaled by fps)."""
    # displacement per frame is already in cm (pix normalized earlier); convert to cm/s
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    """Behavioral state transitions; bins adjusted so semantics are fps-invariant."""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # Original bins (cm/frame): [-inf, 0.5, 2.0, 5.0, inf]
        # Convert to cm/s by multiplying by fps to keep thresholds consistent across fps.
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
    except Exception:
        pass

    return X

def add_longrange_features(
    X,
    center_x,
    center_y,
    fps,
    long_windows: Optional[list] = None,
    ewm_spans: Optional[list] = None,
    pct_windows: Optional[list] = None
):
    """
    Long-range temporal features (windows & spans scaled by fps).
    Parameters:
      - X: DataFrame to append features to
      - center_x, center_y: pd.Series of coordinates (in cm)
      - fps: frames per second (float)
      - long_windows: list of integer window bases (in frames @30fps) for long moving averages (default [120,240,480])
      - ewm_spans: list of integer spans (in frames @30fps) for EWM (default [60,120,240])
      - pct_windows: list of integer window bases for speed percentile ranking (default [60,120,240])
    """
    if long_windows is None:
        long_windows = [30, 60,120, 240, 480]
    if ewm_spans is None:
        ewm_spans = [15, 30,60, 120, 240]
    if pct_windows is None:
        pct_windows = [15, 30,60, 120, 240]

    # long moving average of positions
    for window in long_windows:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # EWM (span interpreted in frames)
    for span in ewm_spans:
        s = _scale(span, fps)
        # pandas ewm accepts span as float/int; keep min_periods=1 to avoid excessive NaNs
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    # speed-based percentile rank over windows
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in pct_windows:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base: int = 180, colname: str = "path_cum180"):
    L = max(1, _scale(horizon_frames_base, fps))  # frames
    # step length (cm per frame since coords are cm)
    step = np.hypot(cx.diff(), cy.diff())
    # centered rolling sum over ~2L+1 frames (acausal)
    path = step.rolling(2*L + 1, min_periods=max(5, L//6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    return X


def add_groom_microfeatures(X, df, fps):
    parts = df.columns.get_level_values(0)
    if 'body_center' not in parts or 'nose' not in parts:
        return X

    cx = df['body_center']['x']; cy = df['body_center']['y']
    nx = df['nose']['x']; ny = df['nose']['y']

    cs = (np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)).fillna(0)
    ns = (np.sqrt(nx.diff()**2 + ny.diff()**2) * float(fps)).fillna(0)

    w30 = _scale(30, fps)
    X['head_body_decouple'] = (ns / (cs + 1e-3)).clip(0, 10).rolling(w30, min_periods=max(1, w30//3)).median()

    r = np.sqrt((nx - cx)**2 + (ny - cy)**2)
    X['nose_rad_std'] = r.rolling(w30, min_periods=max(1, w30//3)).std().fillna(0)

    if 'tail_base' in parts:
        ang = np.arctan2(df['nose']['y']-df['tail_base']['y'], df['nose']['x']-df['tail_base']['x'])
        dang = np.abs(ang.diff()).fillna(0)
        X['head_orient_jitter'] = dang.rolling(w30, min_periods=max(1, w30//3)).mean()

    return X


def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Social interaction features (windows scaled by fps)."""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # per-frame velocities (cm/frame)
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()  # decreasing distance => positive approach
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X

# ===============================================================
# 1) Past–vs–Future speed asymmetry (acausal, continuous)
#    Δv = mean_future(speed) - mean_past(speed)
# ===============================================================
def add_speed_asymmetry_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    horizon_base: int = 30, agg: str = "mean"
) -> pd.DataFrame:
    w = max(3, _scale(horizon_base, fps))
    v = _speed(cx, cy, fps)
    if agg == "median":
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).median()
        v_fut  = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).median().iloc[::-1]
    else:
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).mean()
        v_fut  = _roll_future_mean(v, w, min_p=max(3, w//4))
    X["spd_asym_1s"] = (v_fut - v_past).fillna(0.0)
    return X

# ===============================================================
# 2) Distribution shift (future vs past) via symmetric KL of
#    Gaussian fits on speed 
# ===============================================================
def add_gauss_shift_speed_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    window_base: int = 30, eps: float = 1e-6
) -> pd.DataFrame:
    w = max(5, _scale(window_base, fps))
    v = _speed(cx, cy, fps)

    mu_p = v.rolling(w, min_periods=max(3, w//4)).mean()
    va_p = v.rolling(w, min_periods=max(3, w//4)).var().clip(lower=eps)

    mu_f = _roll_future_mean(v, w, min_p=max(3, w//4))
    va_f = _roll_future_var(v, w, min_p=max(3, w//4)).clip(lower=eps)

    # KL(Np||Nf) + KL(Nf||Np)
    kl_pf = 0.5 * ((va_p/va_f) + ((mu_f - mu_p)**2)/va_f - 1.0 + np.log(va_f/va_p))
    kl_fp = 0.5 * ((va_f/va_p) + ((mu_p - mu_f)**2)/va_p - 1.0 + np.log(va_p/va_f))
    X["spd_symkl_1s"] = (kl_pf + kl_fp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def add_segmental_features_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    horizons_base=(30, 90), pause_thr_cms: float = 2.0
) -> pd.DataFrame:
    """Segment-level trend/acc/jerk/pause over ~1s/3s windows (fps-aware)."""
    v = _speed(cx, cy, fps)
    acc = v.diff() * float(fps)
    jerk = acc.diff() * float(fps)

    for h in horizons_base:
        w = max(3, _scale(h, fps))
        min_p = max(3, w // 3)

        def _slope(arr):
            arr = arr[np.isfinite(arr)]
            L = len(arr)
            if L < 2:
                return 0.0
            idx = np.arange(L, dtype=np.float32)
            sum_x = float(idx.sum())
            sum_x2 = float((idx * idx).sum())
            denom = L * sum_x2 - sum_x * sum_x
            if denom == 0:
                return 0.0
            sum_y = float(arr.sum())
            sum_xy = float((idx * arr).sum())
            return (L * sum_xy - sum_x * sum_y) / denom

        X[f'sp_sl_{h}'] = v.rolling(w, center=True, min_periods=min_p).apply(_slope, raw=True)
        X[f'acc_m_{h}'] = acc.rolling(w, center=True, min_periods=min_p).mean()
        X[f'acc_v_{h}'] = acc.rolling(w, center=True, min_periods=min_p).var()
        X[f'jerk_m_{h}'] = jerk.rolling(w, center=True, min_periods=min_p).mean()
        X[f'pause_r_{h}'] = (v < pause_thr_cms).rolling(w, center=True, min_periods=1).mean()

    return X


# %%
def transform_single(single_mouse, body_parts_tracked, fps):
    """Enhanced single mouse transform (FPS-aware windows/lags; distances in cm)."""
    available_body_parts = single_mouse.columns.get_level_values(0)

    # Base distance features (squared distances across body parts)
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        speed_lags = [1,2,3,4,5, 10, 20, 30,40,50,60]
        speed_parts = []
        for lag_base in speed_lags:
            lag = _scale(lag_base, fps)
            shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
            suf = f"l{lag_base}"
            speed_parts.append(pd.DataFrame({
                f'sp_lf_{suf}': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
                f'sp_rt_{suf}': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
                f'sp_lf2_{suf}': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
                f'sp_rt2_{suf}': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
            }))

        speeds = pd.concat(speed_parts, axis=1)
        # Keep original names for backward compatibility (based on 10-frame base lag)
        if 'sp_lf_l10' in speeds:
            speeds = speeds.assign(
                sp_lf=speeds['sp_lf_l10'],
                sp_rt=speeds['sp_rt_l10'],
                sp_lf2=speeds['sp_lf2_l10'],
                sp_rt2=speeds['sp_rt2_l10'],
            )
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Body angle (orientation)
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    # Core temporal features (windows scaled by fps)
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            X[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            X[f'cy_s{w}'] = cy.rolling(ws, **roll).std()
            X[f'x_rng{w}'] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f'y_rng{w}'] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

        # Advanced features (fps-scaled)
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)
        X = add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base=180)
        X = add_groom_microfeatures(X, single_mouse, fps)
        X = add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base=30)         
        X = add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base=30)
        X = add_segmental_features_single(X, cx, cy, fps, horizons_base=(30, 90))
  
    # Nose-tail features with duration-aware lags
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    # Ear features with duration-aware offsets
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-20, -10, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)  
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """Enhanced pair transform (FPS-aware windows/lags; distances in cm)."""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # Inter-mouse distances (squared distances across all part pairs)
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        speed_lags = [5, 10, 20, 30]
        speed_parts = []
        for lag_base in speed_lags:
            lag = _scale(lag_base, fps)
            shA = mouse_pair['A']['ear_left'].shift(lag)
            shB = mouse_pair['B']['ear_left'].shift(lag)
            suf = f"l{lag_base}"
            speed_parts.append(pd.DataFrame({
                f'sp_A_{suf}': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
                f'sp_AB_{suf}': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
                f'sp_B_{suf}': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
            }))

        speeds = pd.concat(speed_parts, axis=1)
        if 'sp_A_l10' in speeds:
            speeds = speeds.assign(
                sp_A=speeds['sp_A_l10'],
                sp_AB=speeds['sp_AB_l10'],
                sp_B=speeds['sp_B_l10'],
            )
        X = pd.concat([X, speeds], axis=1)

    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base', 'body_center']) and all(p in avail_B for p in ['nose', 'tail_base', 'body_center']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)
        # Head-to-head alignment (parallel vs antiparallel vs orthogonal)
        head_dot = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y'])
        head_den = (np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)
        head_cos = head_dot / head_den
        X['head_cos'] = head_cos
        X['head_parallel'] = (head_cos > 0.5).astype(float)
        X['head_antipar'] = (head_cos < -0.5).astype(float)
        X['head_side'] = ((head_cos >= -0.5) & (head_cos <= 0.5)).astype(float)

        def _ang(dx, dy):
            return np.arctan2(dy, dx)

        # Nose pointing to opponent body_center (facing vs back/side)
        ang_A_head = _ang(dir_A['x'], dir_A['y'])
        ang_B_head = _ang(dir_B['x'], dir_B['y'])
        ang_A_to_Bc = _ang(mouse_pair['B']['body_center']['x'] - mouse_pair['A']['nose']['x'],
                           mouse_pair['B']['body_center']['y'] - mouse_pair['A']['nose']['y'])
        ang_B_to_Ac = _ang(mouse_pair['A']['body_center']['x'] - mouse_pair['B']['nose']['x'],
                           mouse_pair['A']['body_center']['y'] - mouse_pair['B']['nose']['y'])

        def _wrap_diff(a, b):
            d = a - b
            return np.arctan2(np.sin(d), np.cos(d))

        ang_A_diff = _wrap_diff(ang_A_to_Bc, ang_A_head)
        ang_B_diff = _wrap_diff(ang_B_to_Ac, ang_B_head)

        X['A_face_cos'] = np.cos(ang_A_diff)
        X['B_face_cos'] = np.cos(ang_B_diff)

        # Facing bins (one-hot-ish)
        X['A_face_front'] = (X['A_face_cos'] > 0.7).astype(float)
        X['A_face_back']  = (X['A_face_cos'] < -0.5).astype(float)
        X['A_face_side']  = ((X['A_face_cos'] <= 0.7) & (X['A_face_cos'] >= -0.5)).astype(float)
        X['B_face_front'] = (X['B_face_cos'] > 0.7).astype(float)
        X['B_face_back']  = (X['B_face_cos'] < -0.5).astype(float)
        X['B_face_side']  = ((X['B_face_cos'] <= 0.7) & (X['B_face_cos'] >= -0.5)).astype(float)

        # Ear-to-opponent-nose wedge: captures side-on vs face-on
        if all(p in avail_A for p in ['ear_left', 'ear_right']) and 'nose' in avail_B:
            ang_el = _ang(mouse_pair['B']['nose']['x'] - mouse_pair['A']['ear_left']['x'],
                          mouse_pair['B']['nose']['y'] - mouse_pair['A']['ear_left']['y'])
            ang_er = _ang(mouse_pair['B']['nose']['x'] - mouse_pair['A']['ear_right']['x'],
                          mouse_pair['B']['nose']['y'] - mouse_pair['A']['ear_right']['y'])
            X['A_ear_wedge'] = np.abs(_wrap_diff(ang_el, ang_er))
        if all(p in avail_B for p in ['ear_left', 'ear_right']) and 'nose' in avail_A:
            ang_el = _ang(mouse_pair['A']['nose']['x'] - mouse_pair['B']['ear_left']['x'],
                          mouse_pair['A']['nose']['y'] - mouse_pair['B']['ear_left']['y'])
            ang_er = _ang(mouse_pair['A']['nose']['x'] - mouse_pair['B']['ear_right']['x'],
                          mouse_pair['A']['nose']['y'] - mouse_pair['B']['ear_right']['y'])
            X['B_ear_wedge'] = np.abs(_wrap_diff(ang_el, ang_er))

    # Approach rate (duration-aware lag)
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    # Distance bins (cm; unchanged by fps)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                     (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

    # Temporal interaction features (fps-adjusted windows)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'd_m{w}']  = cd_full.rolling(ws, **roll).mean()
            X[f'd_s{w}']  = cd_full.rolling(ws, **roll).std()
            X[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
            X[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

            d_var = cd_full.rolling(ws, **roll).var()
            X[f'int{w}'] = 1 / (1 + d_var)

            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
            X[f'co_s{w}'] = coord.rolling(ws, **roll).std()

    # Nose-nose dynamics (duration-aware lags)
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [1,2,3,4,5,6,7,8,9,10, 20, 30, 40, 50, 60, 70,80,90,100]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

        # Nose approach vs lateral slip (cm/s; >0 means separating along radial line)
        rel_x = mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x']
        rel_y = mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y']
        Avx = mouse_pair['A']['nose']['x'].diff()
        Avy = mouse_pair['A']['nose']['y'].diff()
        Bvx = mouse_pair['B']['nose']['x'].diff()
        Bvy = mouse_pair['B']['nose']['y'].diff()
        rel_den = (nn + 1e-6)
        rv = ((Avx - Bvx) * rel_x + (Avy - Bvy) * rel_y) / rel_den
        lv = ((Avx - Bvx) * (-rel_y) + (Avy - Bvy) * rel_x) / rel_den
        X['nn_rad_sp'] = (rv * float(fps)).fillna(0)
        X['nn_lat_sp'] = (lv * float(fps)).fillna(0)

        # Nose speeds and imbalance (cm/s)
        A_sp = np.sqrt(Avx**2 + Avy**2) * float(fps)
        B_sp = np.sqrt(Bvx**2 + Bvy**2) * float(fps)
        gap_sp = (A_sp - B_sp)
        X['nose_spdA'] = A_sp
        X['nose_spdB'] = B_sp
        X['nose_spd_gap'] = gap_sp
        for lag in [5, 10, 20, 30]:
            l = _scale(lag, fps)
            X[f'nose_spdA_lg{lag}'] = A_sp.shift(l)
            X[f'nose_spdB_lg{lag}'] = B_sp.shift(l)
            X[f'nose_spd_gap_lg{lag}'] = gap_sp.shift(l)

        # Rolling proximity stats and multi-threshold contact ratios
        roll_opts = dict(min_periods=1, center=True)
        for w in [5, 15, 45]:
            ws = _scale(w, fps)
            X[f'nn_mean{w}'] = nn.rolling(ws, **roll_opts).mean()
            X[f'nn_std{w}']  = nn.rolling(ws, **roll_opts).std()
            for thr in (8.0, 12.0, 15.0):
                X[f'nn_ct{int(thr)}_{w}'] = (nn < thr).rolling(ws, **roll_opts).mean()

    # Velocity alignment (duration-aware offsets)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        for off in [-20, -10, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        # Advanced interaction (fps-adjusted internals)
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)
        

    return X.astype(np.float32, copy=False)


# %%

def tune_threshold(oof_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Search the probability cutoff that maximizes F1."""
    def objective(trial):
        threshold = trial.suggest_float("threshold", 0.0, 1.0, step=0.01)
        return f1_score(y_true, (oof_pred >= threshold), zero_division=0)

    # Silence per-trial Optuna INFO logs; only report when we get a better score.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_val = [-np.inf]

    def _report_best(study, trial):
        if trial.value is None:
            return
        if trial.value > best_val[0]:
            best_val[0] = trial.value
            if verbose:
                print(f"  [tune] trial={trial.number} new best thr={trial.params['threshold']:.2f} f1={trial.value:.4f}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200, n_jobs=-1, callbacks=[_report_best])
    return float(study.best_params["threshold"])


# %%
def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, label, meta, n_samples=1_500_000):
    slug = _slugify(body_parts_tracked_str)
    thr_dir = THRESHOLD_DIR
    mdl_dir = MODEL_DIR
    thr_load_dir = THRESHOLD_LOAD_DIR
    mdl_load_dir = MODEL_LOAD_DIR
    os.makedirs(thr_dir, exist_ok=True)
    os.makedirs(mdl_dir, exist_ok=True)
    thr_path = os.path.join(thr_dir, f"{slug}_{switch_tr}_thresholds.pkl")
    mdl_path = os.path.join(mdl_dir, f"{slug}_{switch_tr}_models.pkl")
    thr_load_path = os.path.join(thr_load_dir, f"{slug}_{switch_tr}_thresholds.pkl")
    mdl_load_path = os.path.join(mdl_load_dir, f"{slug}_{switch_tr}_models.pkl")

    models = []
    xgb0 = _make_xgb(
        n_estimators=180, learning_rate=0.08, max_depth=6,
        min_child_weight=8 if USE_GPU else 5, gamma=1.0 if USE_GPU else 0.,
        subsample=0.8, colsample_bytree=0.8, single_precision_histogram=USE_GPU,
        verbosity=0
    )
    models.append(make_pipeline(StratifiedSubsetClassifier(xgb0, n_samples and int(n_samples/1.2))))

    xgb_250 = _make_xgb(
        n_estimators=250, learning_rate=0.08, max_depth=6,
        min_child_weight=8 if USE_GPU else 5, gamma=1.0 if USE_GPU else 0.,
        subsample=0.8, colsample_bytree=0.8, single_precision_histogram=USE_GPU,
        verbosity=0
    )
    models.append(make_pipeline(StratifiedSubsetClassifier(xgb_250, n_samples and int(n_samples/1.2))))

    model_names = ['xgb_180','xgb_250']

    if USE_GPU:
        # GPU-only heavy XGBs etc (same as before)
        xgb1 = XGBClassifier(
            random_state=SEED, booster="gbtree", tree_method="gpu_hist",
            n_estimators=2000, learning_rate=0.05, grow_policy="lossguide",
            max_leaves=255, max_depth=0, min_child_weight=10, gamma=0.0,
            subsample=0.90, colsample_bytree=1.00, colsample_bylevel=0.85,
            reg_alpha=0.0, reg_lambda=1.0, max_bin=256,
            single_precision_histogram=True, verbosity=0
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(xgb1, n_samples and int(n_samples/2.),
                                            random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                                            es_rounds="auto", es_metric="auto")
        ))
        xgb2 = XGBClassifier(
            random_state=SEED, booster="gbtree", tree_method="gpu_hist",
            n_estimators=1400, learning_rate=0.06, max_depth=7,
            min_child_weight=12, subsample=0.70, colsample_bytree=0.80,
            reg_alpha=0.0, reg_lambda=1.5, max_bin=256,
            single_precision_histogram=True, verbosity=0
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(xgb2, n_samples and int(n_samples/1.5),
                                            random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                                            es_rounds="auto", es_metric="auto")
        ))
        model_names.extend(['xgb1', 'xgb2'])

    action_thresholds = defaultdict(lambda: 0.27)
    model_list = []

    # ---------- thresholds ----------
    if not USE_ADAPTIVE_THRESHOLDS:
        if verbose:
            print(f"[thr] using fixed threshold 0.27 | {switch_tr}")
    elif LOAD_THRESHOLDS and os.path.exists(thr_load_path):
        loaded_thr = joblib.load(thr_load_path)
        action_thresholds.update(loaded_thr)
        if verbose:
            print(f"[thr] loaded thresholds from {thr_load_path}")
    else:
        for action in label.columns:
            action_mask = ~label[action].isna().values
            y_action = label[action][action_mask].values.astype(int)
            meta_masked = meta.iloc[action_mask]
            groups_action = meta_masked.video_id.values

            tune_cap = n_samples
            X_action = X_tr[action_mask]
            X_tune = X_action
            y_tune = y_action
            groups_tune = groups_action
            if len(y_tune) > tune_cap:
                rng = np.random.default_rng(SEED)
                keep_idx = rng.choice(len(y_tune), size=tune_cap, replace=False)
                X_tune = X_action.iloc[keep_idx]
                y_tune = y_action[keep_idx]
                groups_tune = groups_action[keep_idx]

            unique_groups = np.unique(groups_tune)
            n_pos = int(y_tune.sum())
            n_neg = int(len(y_tune) - n_pos)
            if len(unique_groups) >= 2 and n_pos > 0 and n_neg > 0:
                n_splits = min(5, len(unique_groups))
                cv_raw = GroupKFold(n_splits=n_splits)
                splits = list(cv_raw.split(np.zeros(len(y_tune), dtype=np.int8), y_tune, groups_tune))

                single_class_fold = any(
                    (np.unique(y_tune[tr_idx]).size < 2) or (np.unique(y_tune[te_idx]).size < 2)
                    for tr_idx, te_idx in splits
                )
                if single_class_fold:
                    if verbose:
                        print(f"threshold tuning skipped (single-class fold) | {switch_tr} | action={action}")
                else:
                    base_model = clone(models[0])
                    oof_pred = cross_val_predict(
                        base_model,
                        X_tune,
                        y_tune,
                        cv=splits,
                        groups=groups_tune,
                        method="predict_proba",
                        n_jobs=1
                    )[:, 1]
                    tuned_thr = tune_threshold(oof_pred, y_tune)
                    action_thresholds[action] = tuned_thr
                    if verbose:
                        print(f"tuned threshold {tuned_thr:.2f} | {switch_tr} | action={action} | groups={len(unique_groups)}")
                    del base_model, oof_pred, splits
                    gc.collect()
            else:
                if verbose:
                    print(f"threshold tuning skipped (insufficient groups or classes) | {switch_tr} | action={action}")

    try:
        joblib.dump(dict(action_thresholds), thr_path)
        if verbose:
            print(f"saved thresholds -> {thr_path}")
    except OSError as e:
        if verbose:
            print(f"[thr] save failed ({e}); skipping save.")

    if ONLY_TUNE_THRESHOLDS:
        if verbose:
            print("[mode] ONLY_TUNE_THRESHOLDS=True, skipping training and inference.")
        del X_tr; gc.collect()
        return

    # ---------- models ----------
    if LOAD_MODELS and os.path.exists(mdl_load_path):
        model_list = joblib.load(mdl_load_path)
        if verbose:
            print(f"[mdl] loaded models from {mdl_load_path}")
        if CHECK_LOAD:
            # Drop loaded actions that have no positives in current data
            filtered = []
            for action, trained in model_list:
                action_mask = ~label[action].isna().values
                y_action = label[action][action_mask].values.astype(int)
                if y_action.sum() == 0:
                    if verbose:
                        print(f"[mdl] drop loaded action={action} (no positives in current data)")
                    continue
                filtered.append((action, trained))
            model_list = filtered
    else:
        for action in label.columns:
            action_mask = ~label[action].isna().values
            y_action = label[action][action_mask].values.astype(int)
            meta_masked = meta.iloc[action_mask]
            groups_action = meta_masked.video_id.values
            if y_action.sum() == 0:
                if verbose:
                    print(f"[mdl] skip action={action} (no positives)")
                continue

            trained = []
            for model_idx, m in enumerate(models):
                m_clone = clone(m)
                t0 = perf_counter()
                m_clone.fit(X_tr[action_mask], y_action)
                dt = perf_counter() - t0
                print(f"trained model {model_names[model_idx]} | {switch_tr} | action={action} | {dt:.1f}s", flush=True)
                trained.append(m_clone)

            if trained:
                model_list.append((action, trained))

        try:
            joblib.dump(model_list, mdl_path)
            if verbose:
                print(f"saved models -> {mdl_path}")
        except OSError as e:
            if verbose:
                print(f"[mdl] save failed ({e}); skipping save.")

    del X_tr; gc.collect()

    # ---- TEST INFERENCE ----
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )
    fps_lookup = (test_subset[['video_id','frames_per_second']]
                    .drop_duplicates('video_id')
                    .set_index('video_id')['frames_per_second'].to_dict())

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)

            del data_te

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained in model_list:
                if action in actions_te:
                    probs = []
                    for mi, mdl in enumerate(trained):
                        probs.append(mdl.predict_proba(X_te)[:, 1])
                    pred[action] = np.mean(probs, axis=0)

            del X_te; gc.collect()

            if pred.shape[1] != 0:
                submission_list.append(predict_multiclass_adaptive(pred, meta_te, action_thresholds))
        except Exception as e:
            print(e)
            try: del data_te
            except: pass
            gc.collect()

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
# ==================== MAIN LOOP ====================

submission_list = []

for section in range(len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    # try:
    body_parts_tracked = json.loads(body_parts_tracked_str)
    print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

    _fps_lookup = (
        train_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    single_list, single_label_list, single_meta_list = [], [], []
    pair_list, pair_label_list, pair_meta_list = [], [], []

    for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
        if switch == 'single':
            single_list.append(data)
            single_meta_list.append(meta)
            single_label_list.append(label)
        else:
            pair_list.append(data)
            pair_meta_list.append(meta)
            pair_label_list.append(label)

    if len(single_list) > 0:
        single_frame_counts = [len(meta) for meta in single_meta_list]
        total_single_frames = sum(single_frame_counts)
        single_needs_sampling = total_single_frames > N_CUT

        single_feats_parts = []
        for idx, (data_i, meta_i, label_i, frames_i) in enumerate(
            zip(single_list, single_meta_list, single_label_list, single_frame_counts)
        ):
            fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
            Xi_full = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)

            if single_needs_sampling and frames_i > 0:
                sample_n = int(round(N_CUT * (frames_i / total_single_frames)))
                sample_n = min(len(Xi_full), max(1, sample_n))
                # Stratify on "any action active" to keep positive/negative balance
                y_arr = (label_i.sum(axis=1) > 0).astype(int).values
                if verbose:
                    counts = np.bincount(y_arr, minlength=2)
                    # print(f"[debug] single sampling | section={section} | frames={frames_i} | sample_n={sample_n} | counts={counts.tolist()}")
                # Ensure minimum positive share in the sampled set
                rng = np.random.default_rng(SEED)
                pos_idx = np.where(y_arr == 1)[0]
                neg_idx = np.where(y_arr == 0)[0]
                n_pos = len(pos_idx); n_neg = len(neg_idx)
                target_pos = min(n_pos, max(0, int(round(sample_n * LEAST_POS_PERC)))) if n_pos > 0 else 0
                target_neg = sample_n - target_pos
                if np.min(np.bincount(y_arr, minlength=2)) < 2 or sample_n >= len(y_arr):
                    strat_idx = rng.choice(len(y_arr), size=sample_n, replace=False)
                else:
                    # draw positives first (up to target_pos), rest from negatives
                    pos_keep = rng.choice(pos_idx, size=target_pos, replace=False) if target_pos > 0 else np.array([], dtype=int)
                    neg_keep = rng.choice(neg_idx, size=min(target_neg, n_neg), replace=False)
                    strat_idx = np.concatenate([pos_keep, neg_keep])
                    if len(strat_idx) < sample_n:
                        # top up with any remaining samples if needed
                        remain = sample_n - len(strat_idx)
                        pool = np.setdiff1d(np.arange(len(y_arr)), strat_idx, assume_unique=False)
                        extra = rng.choice(pool, size=min(remain, len(pool)), replace=False)
                        strat_idx = np.concatenate([strat_idx, extra])
                    rng.shuffle(strat_idx)
                Xi = Xi_full.iloc[strat_idx]
                meta_i = meta_i.iloc[strat_idx]
                label_i = label_i.iloc[strat_idx]
            else:
                Xi = Xi_full

            # keep meta/label aligned with sampled features
            single_meta_list[idx] = meta_i
            single_label_list[idx] = label_i
            single_feats_parts.append(Xi)
            del Xi

        X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)

        single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
        single_meta  = pd.concat(single_meta_list,  axis=0, ignore_index=True)

        del single_list, single_label_list, single_meta_list, single_feats_parts
        gc.collect()

        print(f"  Single: {X_tr.shape}")
        submit_ensemble(body_parts_tracked_str, 'single', X_tr, single_label, single_meta)

        del X_tr, single_label, single_meta
        gc.collect()

    if len(pair_list) > 0:
        pair_frame_counts = [len(meta) for meta in pair_meta_list]
        total_pair_frames = sum(pair_frame_counts)
        pair_needs_sampling = total_pair_frames > N_CUT

        pair_feats_parts = []
        for idx, (data_i, meta_i, label_i, frames_i) in enumerate(
            zip(pair_list, pair_meta_list, pair_label_list, pair_frame_counts)
        ):
            fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
            Xi_full = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)

            if pair_needs_sampling and frames_i > 0:
                sample_n = int(round(N_CUT * (frames_i / total_pair_frames)))
                sample_n = min(len(Xi_full), max(1, sample_n))
                # Stratify on "any action active" to keep positive/negative balance
                y_arr = (label_i.sum(axis=1) > 0).astype(int).values
                if verbose:
                    counts = np.bincount(y_arr, minlength=2)
                    # print(f"[debug] pair sampling | section={section} | frames={frames_i} | sample_n={sample_n} | counts={counts.tolist()}")
                rng = np.random.default_rng(SEED)
                pos_idx = np.where(y_arr == 1)[0]
                neg_idx = np.where(y_arr == 0)[0]
                n_pos = len(pos_idx); n_neg = len(neg_idx)
                target_pos = min(n_pos, max(0, int(round(sample_n * LEAST_POS_PERC)))) if n_pos > 0 else 0
                target_neg = sample_n - target_pos
                if np.min(np.bincount(y_arr, minlength=2)) < 2 or sample_n >= len(y_arr):
                    strat_idx = rng.choice(len(y_arr), size=sample_n, replace=False)
                else:
                    pos_keep = rng.choice(pos_idx, size=target_pos, replace=False) if target_pos > 0 else np.array([], dtype=int)
                    neg_keep = rng.choice(neg_idx, size=min(target_neg, n_neg), replace=False)
                    strat_idx = np.concatenate([pos_keep, neg_keep])
                    if len(strat_idx) < sample_n:
                        remain = sample_n - len(strat_idx)
                        pool = np.setdiff1d(np.arange(len(y_arr)), strat_idx, assume_unique=False)
                        extra = rng.choice(pool, size=min(remain, len(pool)), replace=False)
                        strat_idx = np.concatenate([strat_idx, extra])
                    rng.shuffle(strat_idx)
                Xi = Xi_full.iloc[strat_idx]
                meta_i = meta_i.iloc[strat_idx]
                label_i = label_i.iloc[strat_idx]
            else:
                Xi = Xi_full

            pair_meta_list[idx] = meta_i
            pair_label_list[idx] = label_i
            pair_feats_parts.append(Xi)
            del Xi

        X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)

        
        pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)
        pair_meta  = pd.concat(pair_meta_list,  axis=0, ignore_index=True)

        del pair_list, pair_label_list, pair_meta_list, pair_feats_parts
        gc.collect()

        print(f"  Pair: {X_tr.shape}")
        submit_ensemble(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta)

        del X_tr, pair_label, pair_meta
        gc.collect()

    # except Exception as e:
    #     print(f'***Exception*** {str(e)}')

    gc.collect()
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

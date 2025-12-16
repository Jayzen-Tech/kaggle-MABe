# kaggle-MABe

## English Version

### Directory Layout & Required Data
The project uses `/kaggle/kaggle-MABe` as the working directory and depends on external Kaggle inputs. Core folders:

```
/kaggle/kaggle-MABe/
├── mabe-final.py                # Final solution script
├── baseline/mabe-baseline.py    # Initial baseline
├── README.md, requirements.txt
└── __pycache__/
```

External data and pretrained artifacts live under `/kaggle/input`, primarily:

```
/kaggle/input/
└── MABe-mouse-behavior-detection/        # Official train/test CSVs and tracking parquet files
```

If you plan to reuse pretrained models/thresholds through `LOAD_MODELS=True` or `LOAD_THRESHOLDS=True`, keep the default folder structure or update the `MODEL_LOAD_DIR` / `THRESHOLD_LOAD_DIR` constants inside the script before running.

### How to Run & Switches

#### Basic Commands
1. (Optional on first run) Install dependencies: `pip install -r requirements.txt`
2. Execute the pipeline: `python mabe-final.py`

Runtime switches at the top of `mabe-final.py` control training/inference behavior:

```python
# ---- runtime switches ----
ONLY_TUNE_THRESHOLDS = False    # True: tune thresholds only and exit (skip training/inference/submission)
USE_ADAPTIVE_THRESHOLDS = True  # False: fallback to a fixed 0.27 threshold for every action
LOAD_THRESHOLDS = False         # True: read thresholds from THRESHOLD_DIR instead of tuning
LOAD_MODELS = False             # True: load pretrained models from MODEL_DIR
CHECK_LOAD = False              # True: after loading, drop actions without positives in the current folds
THRESHOLD_DIR = "./models/threshold"
MODEL_DIR = "./models"
THRESHOLD_LOAD_DIR = "/kaggle/input/xgb-models-new/other/xgb-models-new/13/threshold"
MODEL_LOAD_DIR = "/kaggle/input/xgb-models-new/other/xgb-models-new/13"
NON_FINITE_LOG = "non_finite_features.log"
```

**Recommended usage:**
- **Full training + inference**: `ONLY_TUNE_THRESHOLDS=False`, `LOAD_MODELS=False`. The script trains, predicts, and saves models/thresholds under `MODEL_DIR` / `THRESHOLD_DIR`.
- **Threshold tuning only**: When models already exist, set `ONLY_TUNE_THRESHOLDS=True` (default). The run stops after tuning and writes fresh pickles into `THRESHOLD_DIR`.
- **Reuse trained models**: Set `LOAD_MODELS=True` (optionally `CHECK_LOAD=True` to verify label availability). The script reads Joblib caches from `MODEL_LOAD_DIR`; update the path if your models live elsewhere.
- **Reuse tuned thresholds**: Flip `LOAD_THRESHOLDS=True` to skip tuning and read `.pkl` files from `THRESHOLD_LOAD_DIR`.
- Disable adaptive thresholds (`USE_ADAPTIVE_THRESHOLDS=False`) to fall back to the global 0.27 cutoff for quick sanity checks.
- `NON_FINITE_LOG` records any NaN/Inf columns encountered during training or inference. Point it to another path or truncate it as needed.

### Feature Engineering Upgrades Overview
`mabe-final.py` delivers a full feature-engineering refresh covering tracking cleanup, meta encoding, third-mouse context, single/pair motion modeling, and consistent train/test handling. Each subsection below highlights the additions versus the baseline and quotes the relevant code (all snippets originate from `mabe-final.py`).

#### 1. Tracking Cleanup & Pixel Normalization
- Reorder the multi-level columns and convert pixels to centimeters using `pix_per_cm_approx`.
- Interpolate every body part, fill in both directions, and drop joints that remain all NaN.
- Run all downstream matrices through `sanitize_features` to replace NaN/Inf with interpolations and per-column medians.

```python
# mabe-final.py (generate_mouse_data)
pvid = vid.pivot(columns=['mouse_id','bodypart'], index='video_frame', values=['x','y'])
pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False)
pvid = pvid.sort_index(axis=1)
pvid = pvid.interpolate(limit_direction='both')
pvid = pvid.fillna(method='bfill').fillna(method='ffill')
all_nan_cols = pvid.columns[pvid.isna().all()]
if len(all_nan_cols) > 0:
    pvid = pvid.drop(columns=all_nan_cols)

# send every feature matrix through sanitize_features
def sanitize_features(X: Optional[pd.DataFrame]) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.interpolate(axis=0, limit_direction='both')
    if X.isna().any().any():
        med = X.median(axis=0)
        X = X.fillna(med)
    if X.isna().any().any():
        X = X.fillna(0.0)
    return X
```

#### 2. Meta Encoding & Feature Augmentation
- `_mk_meta` parses agent/target sex and strain, builds integer codes and same-sex/strain indicators, computes arena area/aspect/shape, and aligns meta rows with frame indices.
- `augment_with_meta_features` injects these meta columns into single/pair feature matrices and uses identical sanitization/logging during inference.

```python
# mabe-final.py (_mk_meta excerpt)
m['agent_sex_code'] = np.float32(agent_sex_code)
m['target_sex_code'] = np.float32(target_sex_code if target_key is not None else agent_sex_code)
m['agent_strain_code'] = np.float32(agent_strain_code)
m['target_strain_code'] = np.float32(target_strain_code if target_key is not None else agent_strain_code)
m['same_sex'] = np.float32(1.0 if (...) else 0.0)
m['same_strain'] = np.float32(1.0 if (...) else 0.0)
m['arena_shape_code'] = np.float32(arena_shape_code)
m['arena_area_cm2'] = np.float32(arena_area)
m['arena_aspect_ratio'] = np.float32(arena_aspect)
m = m.set_index('video_frame')
m['video_frame'] = m.index

def augment_with_meta_features(X: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    meta_aligned = meta.reindex(X.index)
    meta_features = pd.DataFrame(index=X.index)
    for raw, name in [...]:
        _copy(raw, name)
    meta_features = sanitize_features(meta_features)
    return pd.concat([X, meta_features], axis=1)
```

#### 3. Third-Mouse Context (Triadic Relations)
- `_compute_third_party_context` converts every non agent/target mouse into distance stats against agent/target/midpoint, contact counts, weaving ratios, and cosine directionality.
- `transform_pair` flattens the CTX MultiIndex and adds ratios (for example `ctx_third_min_dist_*_ratio`) so the model learns how third-party interference modulates behaviors.

```python
def _compute_third_party_context(pvid: pd.DataFrame, agent_label, target_label) -> Optional[pd.DataFrame]:
    ctx = pd.DataFrame(index=agent_bc.index)
    ctx['third_min_dist_agent'] = df_a.min(axis=1)
    ctx['third_mean_dist_agent'] = df_a.mean(axis=1)
    ctx['third_close_agent_cnt'] = (df_a < close_thresh).sum(axis=1)
    ctx['third_close_any'] = (close_any > 0).astype(float)
    ctx['third_between_ratio'] = (df_a + df_b).min(axis=1) / (dist_at + 1e-6)
    ctx['third_cos_agent_closest'] = closest
    ctx.columns = pd.MultiIndex.from_arrays([['CTX'] * len(ctx_cols), ctx_cols, ['value'] * len(ctx_cols)])
    return ctx

# transform_pair excerpt
if 'CTX' in mouse_pair.columns.get_level_values(0):
    ctx_flat = mouse_pair['CTX'].copy()
    ...
    X[col] = ctx_flat[col]
if cd is not None:
    denom = (cd + 1e-6)
    X['ctx_third_min_dist_agent_ratio'] = X['ctx_third_min_dist_agent'] / denom
```

#### 4. Single-Mouse Advanced Motion (Long Horizon + Past/Future)
- When `body_center` is available, `transform_single` composes `add_longrange_features`, curvature, future-vs-past speed asymmetry, Gaussian-shift comparisons, and segment-level metrics.
- `add_longrange_features` mixes configurable windows/EWM/percentiles, and the added operators cover future/past velocity contrasts, KL-style speed shifts, and segment-wise acceleration/jerk/pause ratios.

```python
def add_longrange_features(X, center_x, center_y, fps,
                          long_windows=None, ewm_spans=None, pct_windows=None):
    if long_windows is None:
        long_windows = [30, 60, 120, 240, 480]
    if ewm_spans is None:
        ewm_spans = [15, 30, 60, 120, 240]
    for window in long_windows:
        ws = _scale(window, fps)
        X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
    for span in ewm_spans:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()

def transform_single(...):
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']; cy = single_mouse['body_center']['y']
        ...
        X = add_longrange_features(X, cx, cy, fps)
        X = add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base=180)
        X = add_groom_microfeatures(X, single_mouse, fps)
        X = add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base=30)
        X = add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base=30)
        X = add_segmental_features_single(X, cx, cy, fps, horizons_base=(30, 90))
```

- **Multi-scale speed & lags**: `transform_single` now iterates `speed_lags = [1,2,3,4,5,10,20,30,40,50,60]`, computing squared displacements for `ear_left/right` and `tail_base` (see `mabe-final.py:1194-1225`). `_scale(lag_base, fps)` converts each lag to a real time span, covering both 1–2 frame micro stops and >2 s runs. To maintain backward compatibility, the 10-frame outputs remain exposed as `sp_lf`, `sp_rt`, etc.
- **Nose–tail lagged deltas**: `nt_lg{10,20,40}` and `nt_df{10,20,40}` reuse the FPS-aware shifts (`mabe-final.py:1266-1274`), so stretch/contract cues stay stable across frame rates—coverage the baseline lacked.

#### 5. Pair Interaction Enhancements + Context Ratios
- `transform_pair` enriches lagged inter-mouse speeds, head orientation bins, nose/ear wedge angles, segmented distance stats, radial/tangential nose velocities, rolling contact ratios, and velocity alignment offsets.
- CTX metrics are normalized by pairwise distance to remove raw-scale bias.

```python
def transform_pair(mouse_pair, body_parts_tracked, fps):
    # head alignment & facing bins
    X['head_parallel'] = (head_cos > 0.5).astype(float)
    X['A_face_front'] = (X['A_face_cos'] > 0.7).astype(float)
    ...
    # nose dynamics
    X['nn_rad_sp'] = (rv * float(fps)).fillna(0)
    X['nn_lat_sp'] = (lv * float(fps)).fillna(0)
    X[f'nn_ct{int(thr)}_{w}'] = (nn < thr).rolling(ws, **roll_opts).mean()
    # velocity alignment offsets
    for off in [-20, -10, 10, 20]:
        o = _scale_signed(off, fps)
        X[f'va_{off}'] = val.shift(-o)
    # CTX ratios
    X['ctx_third_min_dist_agent_ratio'] = X['ctx_third_min_dist_agent'] / (cd + 1e-6)
```

- **Multi-lag motion & approach cues**: Pair features reuse `speed_lags = [1,2,3,4,5,10,20,30,40,50,60]` (`mabe-final.py:1302-1325`) to output `sp_A_l{lag}`, `sp_AB_l{lag}`, `sp_B_l{lag}`, separating agent, target, and cross-body displacement intensity over short/long horizons. Nose-to-nose distance now spans lag=1–100 via `nn_lg*`, `nn_ch*`, `cl_ps*`, and `nose_spd*_lg*` (`mabe-final.py:1430-1466`), capturing “approach → brief contact → disengage” rhythms that the 10/20/40-frame baseline missed.

#### 6. Train/Test Consistency & Non-finite Logging
- Training and inference both apply `augment_with_meta_features` + `sanitize_features`, and `log_non_finite_features` records any NaN/Inf columns together with their runtime context.
- Sampling keeps meta/labels aligned with feature slices; inference reindexes by `feature_columns` and zero-fills unseen feature names.

```python
# training stage
Xi_full = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
Xi_full = augment_with_meta_features(Xi_full, meta_i).astype(np.float32, copy=False)
log_non_finite_features(
    Xi_full,
    f"train|single|section={section}|n_parts={len(body_parts_tracked)}|{_meta_summary(meta_i)}"
)

# inference stage
X_te = transform_pair(data_te, body_parts_tracked, fps_i)
X_te = augment_with_meta_features(X_te, meta_te)
if feature_columns:
    X_te = (X_te.reindex(columns=feature_columns, fill_value=0.0)
                .astype(np.float32, copy=False))
log_non_finite_features(X_te, f"test|{switch_te}|...")
```

#### 7. Adaptive Threshold Tuning & Inference Strategy
- `submit_ensemble` searches per-action thresholds on GroupKFold OOF predictions via Optuna and persists the Joblib dumps into `THRESHOLD_DIR` for reuse.
- `predict_multiclass_adaptive` smooths predictions, masks them by tuned thresholds, argmaxes valid classes, and smooths segments again—more robust to class imbalance than the fixed 0.27 baseline.

```python
# submit_ensemble excerpt
for action in label.columns:
    ...
    oof_pred = cross_val_predict(
        base_model, X_tune, y_tune, cv=splits,
        groups=groups_tune, method="predict_proba"
    )[:, 1]
    tuned_thr = tune_threshold(oof_pred, y_tune)
    action_thresholds[action] = tuned_thr
joblib.dump(dict(action_thresholds), thr_path)

# predict_multiclass_adaptive
pred_smoothed = pred.rolling(window=7, min_periods=1, center=True).mean()
thresholds = np.array([action_thresholds.get(action, 0.27) for action in pred_smoothed.columns])
valid = pred_smoothed.values >= thresholds
masked = np.where(valid, pred_smoothed.values, -np.inf)
ama = masked.argmax(axis=1)
all_invalid = ~valid.any(axis=1)
ama[all_invalid] = -1
```

Each upgrade is either missing or heavily simplified in the baseline, and together they explain most of the final solution’s lift. Clear snippets and descriptions make it easier to audit, reproduce, or extend the approach.

## 中文版

## 目录结构与依赖数据
项目以 `/kaggle/kaggle-MABe` 为工作目录，并依赖外部输入。核心目录结构如下：

```
/kaggle/kaggle-MABe/
├── mabe-final.py                # 最终方案脚本
├── baseline/mabe-baseline.py    # 起始 baseline
├── README.md, requirements.txt
└── __pycache__/
```

外部数据与预训练资源位于 `/kaggle/input`，主要包含：

```
/kaggle/input/
└── MABe-mouse-behavior-detection/        # 官方训练/测试 CSV 与追踪 parquet
```

若需通过 `LOAD_MODELS=True` 或 `LOAD_THRESHOLDS=True` 直接复用已训练结果，请确保上述目录保持默认结构，或在运行前修改脚本中的 `MODEL_LOAD_DIR`、`THRESHOLD_LOAD_DIR` 指向新的输入路径。

## 运行方式与参数开关

### 基本命令
1. （首次运行可选）安装依赖：`pip install -r requirements.txt`
2. 执行脚本：`python mabe-final.py`

脚本内部通过一组 runtime switches 控制训练 / 推理流程。它们位于 `mabe-final.py` 顶部，可按当前实验需求修改：

```python
# ---- runtime switches ----
ONLY_TUNE_THRESHOLDS = False    # True: 仅搜索阈值并保存；跳过训练/推理/提交
USE_ADAPTIVE_THRESHOLDS = True  # False: 所有动作使用固定 0.27 阈值
LOAD_THRESHOLDS = False         # True: 从 THRESHOLD_DIR 读取现有阈值，跳过调参
LOAD_MODELS = False             # True: 从 MODEL_DIR 读取已训练模型
CHECK_LOAD = False              # True: 载入模型后过滤掉当前训练集中无正样本的动作
THRESHOLD_DIR = "./models/threshold"
MODEL_DIR = "./models"
THRESHOLD_LOAD_DIR = "/kaggle/input/xgb-models-new/other/xgb-models-new/13/threshold"
MODEL_LOAD_DIR = "/kaggle/input/xgb-models-new/other/xgb-models-new/13"
NON_FINITE_LOG = "non_finite_features.log"
```

**使用建议：**
- **全量训练+推理**：将 `ONLY_TUNE_THRESHOLDS=False`，`LOAD_MODELS=False`，脚本会先训练再预测，并把模型/阈值保存到 `MODEL_DIR`、`THRESHOLD_DIR`。
- **仅调阈值**：当模型已训练好，仅希望在当前折数上重新搜索阈值时，把 `ONLY_TUNE_THRESHOLDS=True`（默认），执行会停止在阈值阶段并把结果写入 `THRESHOLD_DIR`。
- **复用既有模型**：设置 `LOAD_MODELS=True`（必要时 `CHECK_LOAD=True` 验证数据兼容），脚本直接从 `MODEL_LOAD_DIR` 读取 `joblib` 缓存；若文件在自定义位置，请同步更新 `MODEL_LOAD_DIR`。
- **复用阈值**：将 `LOAD_THRESHOLDS=True`，脚本会跳过调参并读取 `THRESHOLD_LOAD_DIR` 中的 `.pkl` 阈值文件。
- `USE_ADAPTIVE_THRESHOLDS=False` 时会退回到全局 0.27 阈值，适用于快速 sanity check。
- `NON_FINITE_LOG` 指向的日志文件记录任何训练/推理特征中出现的 NaN/Inf，便于检测数据问题，可根据需要更改路径或清空。

## 最终方案特征工程改进总览
`mabe-final.py` 针对特征工程进行了系统性升级，覆盖数据清洗、元信息编码、三鼠上下文、单鼠/双鼠运动模式以及训练推理阶段的一致化。本节按照改动类别逐一说明，并附上核心代码片段（均节选自 `mabe-final.py`，仅展示与 baseline 相比新增或改写的部分）。

### 1. 追踪数据清洗与像素归一化
- 重新排序多级列索引、基于 `pix_per_cm_approx` 将像素坐标统一到厘米尺度。
- 对每个 body part 进行插值、双向填充，彻底删除全 NaN 的关节点列。
- 为所有下游特征调用 `sanitize_features`，用插值+中位数填补非有限值。

```python
# mabe-final.py (generate_mouse_data)
pvid = vid.pivot(columns=['mouse_id','bodypart'], index='video_frame', values=['x','y'])
pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False)
pvid = pvid.sort_index(axis=1)
pvid = pvid.interpolate(limit_direction='both')
pvid = pvid.fillna(method='bfill').fillna(method='ffill')
all_nan_cols = pvid.columns[pvid.isna().all()]
if len(all_nan_cols) > 0:
    pvid = pvid.drop(columns=all_nan_cols)

# 全量特征结果进入 sanitize_features
def sanitize_features(X: Optional[pd.DataFrame]) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.interpolate(axis=0, limit_direction='both')
    if X.isna().any().any():
        med = X.median(axis=0)
        X = X.fillna(med)
    if X.isna().any().any():
        X = X.fillna(0.0)
    return X
```

### 2. 元属性编码与 Meta 特征拼接
- `_mk_meta` 解析 agent/target 在原始标注中的性别、品系，生成整型编码与同型/同性指示，额外计算场地面积、纵横比、shape code，并将 meta 与帧号对齐。
- `augment_with_meta_features` 在单鼠/双鼠特征矩阵中注入上述 meta 字段，推理阶段也会使用相同的列顺序与异常值日志。

```python
# mabe-final.py (_mk_meta 片段)
m['agent_sex_code'] = np.float32(agent_sex_code)
m['target_sex_code'] = np.float32(target_sex_code if target_key is not None else agent_sex_code)
m['agent_strain_code'] = np.float32(agent_strain_code)
m['target_strain_code'] = np.float32(target_strain_code if target_key is not None else agent_strain_code)
m['same_sex'] = np.float32(1.0 if (...) else 0.0)
m['same_strain'] = np.float32(1.0 if (...) else 0.0)
m['arena_shape_code'] = np.float32(arena_shape_code)
m['arena_area_cm2'] = np.float32(arena_area)
m['arena_aspect_ratio'] = np.float32(arena_aspect)
m = m.set_index('video_frame')
m['video_frame'] = m.index

def augment_with_meta_features(X: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    meta_aligned = meta.reindex(X.index)
    meta_features = pd.DataFrame(index=X.index)
    for raw, name in [...]:
        _copy(raw, name)  # 按列复制到 float32
    meta_features = sanitize_features(meta_features)
    return pd.concat([X, meta_features], axis=1)
```

### 3. 第三只小鼠上下文特征（三体关系）
- `_compute_third_party_context` 将除 agent/target 外的其他鼠体中心轨迹转换为与 agent、target、midpoint 的距离统计、靠近次数、穿插比例以及余弦方向。
- `transform_pair` 在拼接 CTX MultiIndex 后，进一步构造 `ctx_third_min_dist_*_ratio` 等相对距离特征，帮助模型识别第三方参与对某些行为的影响。

```python
def _compute_third_party_context(pvid: pd.DataFrame, agent_label, target_label) -> Optional[pd.DataFrame]:
    ctx = pd.DataFrame(index=agent_bc.index)
    ctx['third_min_dist_agent'] = df_a.min(axis=1)
    ctx['third_mean_dist_agent'] = df_a.mean(axis=1)
    ctx['third_close_agent_cnt'] = (df_a < close_thresh).sum(axis=1)
    ctx['third_close_any'] = (close_any > 0).astype(float)
    ctx['third_between_ratio'] = (df_a + df_b).min(axis=1) / (dist_at + 1e-6)
    ctx['third_cos_agent_closest'] = closest
    ctx.columns = pd.MultiIndex.from_arrays([['CTX'] * len(ctx_cols), ctx_cols, ['value'] * len(ctx_cols)])
    return ctx

# transform_pair 片段
if 'CTX' in mouse_pair.columns.get_level_values(0):
    ctx_flat = mouse_pair['CTX'].copy()
    ...
    X[col] = ctx_flat[col]
if cd is not None:
    denom = (cd + 1e-6)
    X['ctx_third_min_dist_agent_ratio'] = X['ctx_third_min_dist_agent'] / denom
```

### 4. 单鼠高级运动模式（长区间 + 未来/过去对比）
- `transform_single` 在 body_center 存在时，串联调用 `add_longrange_features`、`add_curvature_features`、`add_speed_asymmetry_future_past_single`、`add_gauss_shift_speed_future_past_single`、`add_segmental_features_single` 等多尺度算子。
- 新的 `add_longrange_features` 支持自定义窗口/EWM/分位数组合，速度未来-过去非对称、速度分布 KL、段级别加速度/jerk/pause 比例等均为新特征。

```python
def add_longrange_features(X, center_x, center_y, fps,
                          long_windows=None, ewm_spans=None, pct_windows=None):
    if long_windows is None:
        long_windows = [30, 60, 120, 240, 480]
    if ewm_spans is None:
        ewm_spans = [15, 30, 60, 120, 240]
    for window in long_windows:
        ws = _scale(window, fps)
        X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
    for span in ewm_spans:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()

def transform_single(...):
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']; cy = single_mouse['body_center']['y']
        ...
        X = add_longrange_features(X, cx, cy, fps)
        X = add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base=180)
        X = add_groom_microfeatures(X, single_mouse, fps)
        X = add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base=30)
        X = add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base=30)
        X = add_segmental_features_single(X, cx, cy, fps, horizons_base=(30, 90))
```

- **多尺度速度与滞后**：`transform_single` 现在遍历 `speed_lags = [1,2,3,4,5,10,20,30,40,50,60]`，对 `ear_left/right` 与 `tail_base` 组合分别计算平方位移（`mabe-final.py:1194-1225`），lag 会通过 `_scale(lag_base, fps)` 转换成统一的时间长度，兼顾 1～2 帧的瞬时急停与 2 秒以上的持续奔跑。为了兼容已有流水线，还保留了 10 帧 lag 的 `sp_lf`、`sp_rt` 等旧列名。
- **鼻尾距离的 lag 差分**：`nt_lg{10,20,40}` 与 `nt_df{10,20,40}` 全量复用 fps 感知 shift（`mabe-final.py:1266-1274`），可在任意帧率下稳定观测体长拉伸与快速蜷缩，这些细节在 baseline 中尚未覆盖。

### 5. 双鼠互动拓展 + 上下文比值
- `transform_pair` 增强了 inter-mouse lagged speeds、头部朝向、鼻尖/耳朵楔形角度、分段距离统计、鼻尖速度径向/切向拆解、滚动接触比例以及速度对齐等特征。
- 对所有 CTX 指标提供相对距离比，避免绝对尺度对行为判断的噪声。

```python
def transform_pair(mouse_pair, body_parts_tracked, fps):
    # head alignment & facing bins
    X['head_parallel'] = (head_cos > 0.5).astype(float)
    X['A_face_front'] = (X['A_face_cos'] > 0.7).astype(float)
    ...
    # nose dynamics
    X['nn_rad_sp'] = (rv * float(fps)).fillna(0)
    X['nn_lat_sp'] = (lv * float(fps)).fillna(0)
    X[f'nn_ct{int(thr)}_{w}'] = (nn < thr).rolling(ws, **roll_opts).mean()
    # velocity alignment offsets
    for off in [-20, -10, 10, 20]:
        o = _scale_signed(off, fps)
        X[f'va_{off}'] = val.shift(-o)
    # CTX 比值
    X['ctx_third_min_dist_agent_ratio'] = X['ctx_third_min_dist_agent'] / (cd + 1e-6)
```

- **双鼠速度/接近的多 lag 描述**：pair 特征同样扩展为 `speed_lags = [1,2,3,4,5,10,20,30,40,50,60]`（`mabe-final.py:1302-1325`），输出 `sp_A_l{lag}`、`sp_AB_l{lag}`、`sp_B_l{lag}`，把 agent、target 以及交叉位移的短期与长期强度拆开；鼻尖距离则覆盖 lag=1~100 的 `nn_lg*`、`nn_ch*`、`cl_ps*` 与 `nose_spd*_lg*`（`mabe-final.py:1430-1466`），能够检测“快速接近→短暂停留→离开”等节奏。相比 baseline 只有 10/20/40 帧，这些多尺度 lag 提供了更丰富的速度与交互时序信息。

### 6. 训练/推理阶段特征一致性与异常监控
- 训练/测试阶段统一使用 `augment_with_meta_features`、`sanitize_features`，并通过 `log_non_finite_features` 记录任何非有限值及上下文，便于排查数据漂移。
- Sampling 阶段保证 meta/label 与特征同步切片，推理阶段使用 `feature_columns` 对齐列顺序并填充缺失特征为 0。

```python
# 训练阶段
Xi_full = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
Xi_full = augment_with_meta_features(Xi_full, meta_i).astype(np.float32, copy=False)
log_non_finite_features(
    Xi_full,
    f"train|single|section={section}|n_parts={len(body_parts_tracked)}|{_meta_summary(meta_i)}"
)

# 推理阶段
X_te = transform_pair(data_te, body_parts_tracked, fps_i)
X_te = augment_with_meta_features(X_te, meta_te)
if feature_columns:
    X_te = (X_te.reindex(columns=feature_columns, fill_value=0.0)
                .astype(np.float32, copy=False))
log_non_finite_features(X_te, f"test|{switch_te}|...")
```

### 7. 自适应阈值寻优与推理策略
- `submit_ensemble` 针对每个动作在 GroupKFold OOF 结果上使用 Optuna 搜索最优概率阈值，若数据量不足则跳过，以 `joblib` 缓存到 `THRESHOLD_DIR` 便于复用。
- `predict_multiclass_adaptive` 先对平滑结果进行逐动作阈值过滤，再取 argmax 并平滑区段，较 baseline 的固定 0.27 更能适应类别不平衡。

```python
# submit_ensemble 片段
for action in label.columns:
    ...
    oof_pred = cross_val_predict(
        base_model, X_tune, y_tune, cv=splits,
        groups=groups_tune, method="predict_proba"
    )[:, 1]
    tuned_thr = tune_threshold(oof_pred, y_tune)
    action_thresholds[action] = tuned_thr
joblib.dump(dict(action_thresholds), thr_path)

# predict_multiclass_adaptive
pred_smoothed = pred.rolling(window=7, min_periods=1, center=True).mean()
thresholds = np.array([action_thresholds.get(action, 0.27) for action in pred_smoothed.columns])
valid = pred_smoothed.values >= thresholds
masked = np.where(valid, pred_smoothed.values, -np.inf)
ama = masked.argmax(axis=1)
all_invalid = ~valid.any(axis=1)
ama[all_invalid] = -1
```

上述每一项均为 baseline 中不存在或大幅简化的部分，共同构成了最终方案在特征层面的主要增益。通过明确的代码片段和描述，可以快速追溯到实现细节，便于复现或二次开发。

import numpy as np
import pandas as pd
import torch
import warnings
import json
import os
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore")

# Dataset Configuration (spam as demonstration example)
DATASETS = {
    'spam': {
        'path': 'data/spam.csv',
        'numerical_cols': list(range(57)),
        'categorical_cols': [],
        'label_col': 57,
    },
}

CONFIG = {
    'dataset': 'spam',
    'miss_ratios': [0.6],
    'n_rounds': 5,
    'n_folds': 5,
    'seed': 42,

    'cagi_params': {
        'mb_size': 128,
        'p_hint': 0.9,
        'n_clusters': 5,
        'iterations': 5000,
        'alpha': 200.0,
        'beta': 1.5,
        'patience': 500,
        'min_delta': 1e-4,
        'sinkhorn_freq': 1000,
        'cluster_update_freq': 500,
        'cluster_method': 'auto',
    },

    'analog_bits_range': 'zero_one',
    'output_dir': './results',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data Preprocessor
def get_num_bits(num_categories):
    if num_categories <= 1:
        return 1
    return int(np.ceil(np.log2(num_categories)))

def int_to_analog_bits(indices, num_categories, bit_value_range='zero_one'):
    num_bits = get_num_bits(num_categories)
    n_samples = len(indices)
    binary_matrix = np.zeros((n_samples, num_bits), dtype=np.float64)
    for i in range(num_bits):
        binary_matrix[:, num_bits - 1 - i] = (indices >> i) & 1
    if bit_value_range == 'minus_one_one':
        return binary_matrix * 2 - 1
    return binary_matrix

def analog_bits_to_int(analog_bits, num_categories, bit_value_range='zero_one'):
    num_bits = get_num_bits(num_categories)
    n_samples = analog_bits.shape[0]
    if bit_value_range == 'minus_one_one':
        binary_matrix = (analog_bits >= 0).astype(np.int64)
    else:
        binary_matrix = (analog_bits >= 0.5).astype(np.int64)
    indices = np.zeros(n_samples, dtype=np.int64)
    for i in range(num_bits):
        indices += binary_matrix[:, num_bits - 1 - i] * (1 << i)
    return np.clip(indices, 0, num_categories - 1)


class DataPreprocessor:

    def __init__(self, numerical_cols, categorical_cols, analog_bits_range='zero_one'):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.analog_bits_range = analog_bits_range
        self.num_scaler = MinMaxScaler()
        self.cat_encoders = {}
        self.bit_info = []
        self.num_num = len(numerical_cols)
        self.num_cat = len(categorical_cols)
        self.fitted = False

    def fit(self, X):
        if self.num_num > 0:
            self.num_scaler.fit(X[:, self.numerical_cols].astype(np.float64))
        self.bit_info = []
        for col_idx in self.categorical_cols:
            col_data_str = pd.Series(X[:, col_idx]).astype(str)
            categories = sorted(col_data_str.unique())
            num_categories = len(categories)
            num_bits = get_num_bits(num_categories)
            self.cat_encoders[col_idx] = {
                'cat_to_idx': {cat: idx for idx, cat in enumerate(categories)},
                'num_categories': num_categories, 'num_bits': num_bits
            }
            start_idx = sum(info['num_bits'] for info in self.bit_info)
            self.bit_info.append({
                'col_idx': col_idx, 'num_categories': num_categories,
                'num_bits': num_bits, 'start_idx': start_idx
            })
        self.total_cat_bits = sum(info['num_bits'] for info in self.bit_info)
        self.fitted = True
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        num_encoded = self.num_scaler.transform(X[:, self.numerical_cols].astype(np.float64)) if self.num_num > 0 else np.zeros((n_samples, 0))
        if self.num_cat > 0:
            cat_bits_list = []
            for col_idx in self.categorical_cols:
                enc = self.cat_encoders[col_idx]
                col_str = pd.Series(X[:, col_idx]).astype(str)
                col_idx_arr = np.array([enc['cat_to_idx'].get(v, 0) for v in col_str])
                cat_bits_list.append(int_to_analog_bits(col_idx_arr, enc['num_categories'], self.analog_bits_range))
            cat_encoded = np.concatenate(cat_bits_list, axis=1)
        else:
            cat_encoded = np.zeros((n_samples, 0))
        return np.concatenate([num_encoded, cat_encoded], axis=1).astype(np.float64)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def decode_categorical(self, X_encoded):
        if self.num_cat == 0:
            return np.zeros((X_encoded.shape[0], 0), dtype=np.int64)
        cat_bits = X_encoded[:, self.num_num:]
        n_samples = X_encoded.shape[0]
        cat_indices = np.zeros((n_samples, len(self.bit_info)), dtype=np.int64)
        for i, info in enumerate(self.bit_info):
            col_bits = cat_bits[:, info['start_idx']:info['start_idx'] + info['num_bits']]
            cat_indices[:, i] = analog_bits_to_int(col_bits, info['num_categories'], self.analog_bits_range)
        return cat_indices


def generate_mcar_mask(n_samples, num_num, num_cat, miss_ratio, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_features = num_num + num_cat
    mask = (np.random.random((n_samples, n_features)) >= miss_ratio).astype(np.float32)
    for i in range(n_samples):
        if mask[i].sum() == 0:
            mask[i, np.random.randint(n_features)] = 1
    return mask


def extend_mask_to_encoded_space(mask_original, num_num, bit_info):
    n_samples = mask_original.shape[0]
    num_mask = mask_original[:, :num_num]
    if len(bit_info) == 0:
        return num_mask
    total_cat_bits = sum(info['num_bits'] for info in bit_info)
    cat_mask_ext = np.zeros((n_samples, total_cat_bits), dtype=np.float32)
    for col_idx, info in enumerate(bit_info):
        cat_col_mask = mask_original[:, num_num + col_idx]
        for b in range(info['num_bits']):
            cat_mask_ext[:, info['start_idx'] + b] = cat_col_mask
    return np.concatenate([num_mask, cat_mask_ext], axis=1)


# Evaluation Metrics
def compute_rmse(X_imputed, X_true, mask_extended, num_num):
    if num_num == 0:
        return np.nan
    missing = (mask_extended[:, :num_num] == 0)
    if np.sum(missing) == 0:
        return 0.0
    return float(np.sqrt(np.mean((X_imputed[:, :num_num][missing] - X_true[:, :num_num][missing]) ** 2)))

def compute_pfc(X_imputed, X_true, mask_original, num_num, preprocessor):
    if preprocessor.num_cat == 0:
        return np.nan
    pred_cat = preprocessor.decode_categorical(X_imputed)
    true_cat = preprocessor.decode_categorical(X_true)
    cat_missing = (mask_original[:, num_num:] == 0)
    total = np.sum(cat_missing)
    if total == 0:
        return 0.0
    return float(np.sum((pred_cat != true_cat) & cat_missing)) / float(total)


# Main
def run_experiment(dataset_name, config):
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print("=" * 80)

    ds_config = DATASETS[dataset_name]
    df = pd.read_csv(ds_config['path'], header=None)
    df.dropna(inplace=True)
    print(f"Data shape: {df.shape}")

    label_col = ds_config['label_col']
    le = LabelEncoder()
    y_encoded = le.fit_transform(df.iloc[:, label_col].values.astype(str))

    feature_cols = [i for i in range(df.shape[1]) if i != label_col]
    X_df = df.iloc[:, feature_cols]

    num_cols_new, cat_cols_new = [], []
    for i, orig_idx in enumerate(feature_cols):
        if orig_idx in ds_config['numerical_cols']:
            num_cols_new.append(i)
        elif orig_idx in ds_config['categorical_cols']:
            cat_cols_new.append(i)

    preprocessor = DataPreprocessor(num_cols_new, cat_cols_new, config['analog_bits_range'])
    X_encoded = preprocessor.fit_transform(X_df.values)
    n_samples = X_encoded.shape[0]
    print(f"Features: {len(num_cols_new)} num + {len(cat_cols_new)} cat")

    cagi_params = config['cagi_params'].copy()
    cagi_params['hidden_dim'] = X_encoded.shape[1] * 2
    cagi_params['numerical_indices'] = list(range(preprocessor.num_num))
    cagi_params['categorical_indices'] = list(range(preprocessor.num_num, X_encoded.shape[1]))

    results = {'dataset': dataset_name, 'results_by_miss_ratio': {}}

    from CAGI import train_cagi, test_cagi

    for miss_ratio in config['miss_ratios']:
        print(f"\nMissing rate: {miss_ratio * 100:.0f}%")
        all_results = {'rmse': [], 'pfc': []}

        for round_idx in range(config.get('n_rounds', 5)):
            round_seed = config['seed'] + round_idx * 1000
            print(f"***** Round {round_idx} *****")
            try:
                kf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=round_seed)
                fold_iter = list(kf.split(X_encoded, y_encoded))
            except:
                kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=round_seed)
                fold_iter = list(kf.split(X_encoded))

            for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
                print(f" Fold {fold_idx +1} / {len(fold_iter)}")
                seed_miss = round_seed + fold_idx * 100 + int(miss_ratio * 10)
                mask_orig = generate_mcar_mask(n_samples, preprocessor.num_num, preprocessor.num_cat, miss_ratio, seed_miss)
                mask_ext = extend_mask_to_encoded_space(mask_orig, preprocessor.num_num, preprocessor.bit_info)

                X_train, X_test = X_encoded[train_idx].copy(), X_encoded[test_idx].copy()
                X_test_true = X_test.copy()

                X_train[mask_ext[train_idx] == 0] = 0.0
                X_test[mask_ext[test_idx] == 0] = 0.0

                try:
                    G, cm = train_cagi(X_train, mask_ext[train_idx], cagi_params, device)
                    X_te_imp, _ = test_cagi(G, X_test, mask_ext[test_idx], device, cagi_params['n_clusters'], cm)

                    rmse = compute_rmse(X_te_imp, X_test_true, mask_ext[test_idx], preprocessor.num_num)
                    pfc = compute_pfc(X_te_imp, X_test_true, mask_orig[test_idx], preprocessor.num_num, preprocessor)

                    all_results['rmse'].append(rmse)
                    all_results['pfc'].append(pfc)

                except Exception as e:
                    print(f"  R{round_idx+1}F{fold_idx+1}: Error - {e}")
                    all_results['rmse'].append(np.nan)
                    all_results['pfc'].append(np.nan)

        summary = {}
        for m, vals in all_results.items():
            clean = [v for v in vals if not np.isnan(v)]
            summary[m] = {'mean': float(np.mean(clean)), 'std': float(np.std(clean))} if clean else {'mean': np.nan, 'std': np.nan}

        results['results_by_miss_ratio'][str(miss_ratio)] = summary
        print(f"\n  Final ({len(all_results['rmse'])} folds):")
        for m in ['rmse', 'pfc']:
            if not np.isnan(summary[m]['mean']):
                print(f"    {m.upper():8s}: {summary[m]['mean']:.4f} +/- {summary[m]['std']:.4f}")

    return results


def main():
    config = CONFIG
    os.makedirs(config['output_dir'], exist_ok=True)
    results = run_experiment(config['dataset'], config)
    if results:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(config['output_dir'], f"{config['dataset']}_{ts}.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()

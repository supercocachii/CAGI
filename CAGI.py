
# CAGI Model: Cluster-Aware Generative Imputation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from geomloss import SamplesLoss
from utils import sample_batch_index, binary_sampler
from typing import Optional, Tuple

# clustering
class BaseCluster:

    def __init__(self, n_clusters=5, max_iter=100, random_state=42, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.tol = tol
        self.centers = None
        self.labels_ = None

    def fit(self, X, M):
        raise NotImplementedError

    def predict(self, X, M):
        raise NotImplementedError

class CustomKMeans(BaseCluster):

    def __init__(self, n_clusters=5, max_iter=100, random_state=42, tol=1e-4):
        super().__init__(n_clusters, max_iter, random_state, tol)
        self.center_masks = None

    def fit(self, X, M):
        n_samples, n_features = X.shape

        completeness = np.sum(M, axis=1)
        top_idx = np.argsort(-completeness)[:self.n_clusters]
        self.centers = X[top_idx].copy()
        center_masks = M[top_idx].copy()

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(self.max_iter):
            X_expanded = X[:, np.newaxis, :]
            centers_expanded = self.centers[np.newaxis, :, :]
            M_expanded = M[:, np.newaxis, :]
            center_masks_expanded = center_masks[np.newaxis, :, :]

            mask_intersection = M_expanded * center_masks_expanded
            valid_counts = np.sum(mask_intersection, axis=2)

            diff = (X_expanded - centers_expanded) * mask_intersection
            squared_diff = diff ** 2

            valid_counts_safe = np.maximum(valid_counts, 1e-8)
            dists = np.sqrt(np.sum(squared_diff, axis=2) * n_features / valid_counts_safe)
            dists[valid_counts == 0] = np.inf

            labels = np.argmin(dists, axis=1)

            old_centers = self.centers.copy()

            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) == 0:
                    continue

                members = X[mask]
                member_masks = M[mask]

                masked_members = np.where(member_masks, members, np.nan)
                new_center = np.nanmean(masked_members, axis=0)

                nan_mask = np.isnan(new_center)
                new_center[nan_mask] = self.centers[k][nan_mask]

                self.centers[k] = new_center
                center_masks[k] = np.any(member_masks, axis=0).astype(float)

            center_shift = np.sum((old_centers - self.centers) ** 2)
            if center_shift < self.tol:
                break

        self.labels_ = labels
        self.center_masks = center_masks
        return self

    def predict(self, X, M):
        n_samples, n_features = X.shape

        X_expanded = X[:, np.newaxis, :]
        centers_expanded = self.centers[np.newaxis, :, :]
        M_expanded = M[:, np.newaxis, :]
        center_masks_expanded = self.center_masks[np.newaxis, :, :]

        mask_intersection = M_expanded * center_masks_expanded
        valid_counts = np.sum(mask_intersection, axis=2)

        diff = (X_expanded - centers_expanded) * mask_intersection
        squared_diff = diff ** 2

        valid_counts_safe = np.maximum(valid_counts, 1e-8)
        dists = np.sqrt(np.sum(squared_diff, axis=2) * n_features / valid_counts_safe)
        dists[valid_counts == 0] = np.inf

        return np.argmin(dists, axis=1)

class CustomKModes(BaseCluster):
    
    def __init__(self, n_clusters=5, max_iter=100, random_state=42, tol=1e-4):
        super().__init__(n_clusters, max_iter, random_state, tol)
        self.center_masks = None

    def _hamming_distance(self, X, centers, M, center_masks):
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        n_features = X.shape[1]
        dists = np.zeros((n_samples, n_clusters))

        for k in range(n_clusters):
            mask_intersection = M * center_masks[k]
            valid_counts = np.sum(mask_intersection, axis=1)
            mismatches = np.sum((X != centers[k]) * mask_intersection, axis=1)
            valid_counts_safe = np.maximum(valid_counts, 1e-8)
            dists[:, k] = mismatches * n_features / valid_counts_safe
            dists[valid_counts == 0, k] = np.inf

        return dists

    def _compute_mode(self, X, M):
        n_features = X.shape[1]
        modes = np.zeros(n_features)
        for j in range(n_features):
            observed_mask = M[:, j] == 1
            if np.sum(observed_mask) > 0:
                values = X[observed_mask, j]
                unique, counts = np.unique(values, return_counts=True)
                modes[j] = unique[np.argmax(counts)]
        return modes

    def fit(self, X, M):
        n_samples, n_features = X.shape

        completeness = np.sum(M, axis=1)
        top_idx = np.argsort(-completeness)[:self.n_clusters]
        self.centers = X[top_idx].copy()
        center_masks = M[top_idx].copy()

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(self.max_iter):
            dists = self._hamming_distance(X, self.centers, M, center_masks)
            labels = np.argmin(dists, axis=1)

            old_centers = self.centers.copy()

            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) == 0:
                    continue
                self.centers[k] = self._compute_mode(X[mask], M[mask])
                center_masks[k] = np.any(M[mask], axis=0).astype(float)

            if np.all(old_centers == self.centers):
                break

        self.labels_ = labels
        self.center_masks = center_masks
        return self

    def predict(self, X, M):
        dists = self._hamming_distance(X, self.centers, M, self.center_masks)
        return np.argmin(dists, axis=1)

class CustomKPrototypes(BaseCluster):

    def __init__(self, n_clusters=5, max_iter=100, random_state=42, tol=1e-4,
                 numerical_indices=None, categorical_indices=None, gamma=0.5):
        super().__init__(n_clusters, max_iter, random_state, tol)
        self.numerical_indices = numerical_indices or []
        self.categorical_indices = categorical_indices or []
        self.gamma = gamma
        self.center_masks = None

    def _compute_distance(self, X, centers, M, center_masks):
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        dists = np.zeros((n_samples, n_clusters))

        for k in range(n_clusters):
            dist_numerical = np.zeros(n_samples)
            dist_categorical = np.zeros(n_samples)

            if len(self.numerical_indices) > 0:
                X_num = X[:, self.numerical_indices]
                center_num = centers[k, self.numerical_indices]
                M_num = M[:, self.numerical_indices]
                center_mask_num = center_masks[k, self.numerical_indices]

                mask_intersection = M_num * center_mask_num
                valid_counts = np.sum(mask_intersection, axis=1)
                diff = (X_num - center_num) * mask_intersection
                squared_diff = np.sum(diff ** 2, axis=1)
                valid_counts_safe = np.maximum(valid_counts, 1e-8)
                dist_numerical = np.sqrt(squared_diff * len(self.numerical_indices) / valid_counts_safe)
                dist_numerical[valid_counts == 0] = 0

            if len(self.categorical_indices) > 0:
                X_cat = X[:, self.categorical_indices]
                center_cat = centers[k, self.categorical_indices]
                M_cat = M[:, self.categorical_indices]
                center_mask_cat = center_masks[k, self.categorical_indices]

                mask_intersection = M_cat * center_mask_cat
                valid_counts = np.sum(mask_intersection, axis=1)
                mismatches = np.sum((X_cat != center_cat) * mask_intersection, axis=1)
                valid_counts_safe = np.maximum(valid_counts, 1e-8)
                dist_categorical = mismatches * len(self.categorical_indices) / valid_counts_safe
                dist_categorical[valid_counts == 0] = 0

            dists[:, k] = dist_numerical + self.gamma * dist_categorical

            total_valid = np.zeros(n_samples)
            if len(self.numerical_indices) > 0:
                total_valid += np.sum(M[:, self.numerical_indices] * center_masks[k, self.numerical_indices], axis=1)
            if len(self.categorical_indices) > 0:
                total_valid += np.sum(M[:, self.categorical_indices] * center_masks[k, self.categorical_indices], axis=1)
            dists[total_valid == 0, k] = np.inf

        return dists

    def _update_center(self, X, M, members_mask):
        n_features = X.shape[1]
        center = np.zeros(n_features)
        members = X[members_mask]
        member_masks = M[members_mask]

        for j in self.numerical_indices:
            observed = member_masks[:, j] == 1
            center[j] = np.mean(members[observed, j]) if np.sum(observed) > 0 else 0

        for j in self.categorical_indices:
            observed = member_masks[:, j] == 1
            if np.sum(observed) > 0:
                values = members[observed, j]
                unique, counts = np.unique(values, return_counts=True)
                center[j] = unique[np.argmax(counts)]

        return center

    def fit(self, X, M):
        n_samples, n_features = X.shape

        if len(self.numerical_indices) == 0 and len(self.categorical_indices) == 0:
            self.numerical_indices = list(range(n_features))

        completeness = np.sum(M, axis=1)
        top_idx = np.argsort(-completeness)[:self.n_clusters]
        self.centers = X[top_idx].copy()
        center_masks = M[top_idx].copy()

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(self.max_iter):
            dists = self._compute_distance(X, self.centers, M, center_masks)
            labels = np.argmin(dists, axis=1)

            old_centers = self.centers.copy()

            for k in range(self.n_clusters):
                members_mask = labels == k
                if np.sum(members_mask) == 0:
                    continue
                self.centers[k] = self._update_center(X, M, members_mask)
                center_masks[k] = np.any(M[members_mask], axis=0).astype(float)

            num_shift = 0
            cat_changed = False
            if len(self.numerical_indices) > 0:
                num_shift = np.sum((old_centers[:, self.numerical_indices] -
                                    self.centers[:, self.numerical_indices]) ** 2)
            if len(self.categorical_indices) > 0:
                cat_changed = not np.all(old_centers[:, self.categorical_indices] ==
                                         self.centers[:, self.categorical_indices])

            if num_shift < self.tol and not cat_changed:
                break

        self.labels_ = labels
        self.center_masks = center_masks
        return self

    def predict(self, X, M):
        dists = self._compute_distance(X, self.centers, M, self.center_masks)
        return np.argmin(dists, axis=1)

def get_cluster_model(method='kmeans', n_clusters=5, max_iter=100, random_state=42,
                      numerical_indices=None, categorical_indices=None, gamma=0.5):
    method = method.lower()

    if method == 'kmeans':
        return CustomKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    elif method == 'kmodes':
        return CustomKModes(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    elif method == 'kprototypes':
        return CustomKPrototypes(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state,
                                 numerical_indices=numerical_indices,
                                 categorical_indices=categorical_indices, gamma=gamma)
    elif method == 'auto':
        has_num = numerical_indices is not None and len(numerical_indices) > 0
        has_cat = categorical_indices is not None and len(categorical_indices) > 0
        if has_num and has_cat:
            return CustomKPrototypes(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state,
                                     numerical_indices=numerical_indices,
                                     categorical_indices=categorical_indices, gamma=gamma)
        elif has_cat:
            return CustomKModes(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        else:
            return CustomKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    else:
        raise ValueError(f"Unknown clustering method: {method}. Options: 'kmeans', 'kmodes', 'kprototypes', 'auto'")

# Generator
class ImprovedGenerator(nn.Module):

    def __init__(self, dim, h_dim, n_clusters):
        super(ImprovedGenerator, self).__init__()
        self.fc1 = nn.Linear(dim * 2 + n_clusters, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(h_dim, dim)

    def forward(self, x, m, c):
        inp = torch.cat([x, m, c], dim=1)
        h = self.dropout1(F.relu(self.bn1(self.fc1(inp))))
        h = self.dropout2(F.relu(self.bn2(self.fc2(h))))
        return torch.sigmoid(self.fc3(h))

# Discriminator
class ImprovedDiscriminator(nn.Module):

    def __init__(self, dim, h_dim):
        super(ImprovedDiscriminator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(h_dim, dim)

    def forward(self, x, h):
        inp = torch.cat([x, h], dim=1)
        out = self.dropout1(F.relu(self.bn1(self.fc1(inp))))
        out = self.dropout2(F.relu(self.bn2(self.fc2(out))))
        return torch.sigmoid(self.fc3(out))

def cluster_onehot(indices, labels, n_clusters):
    onehot = np.zeros((len(indices), n_clusters))
    for i, idx in enumerate(indices):
        if isinstance(idx, (list, np.ndarray)):
            idx = idx[0] if len(idx) > 0 else 0
        onehot[i, labels[idx]] = 1
    return torch.from_numpy(onehot).float()


def improved_noise_sampler(X, M, device):
    X_masked = X * M
    sum_vals = torch.sum(X_masked, dim=0, keepdim=True)
    count_vals = torch.clamp(torch.sum(M, dim=0, keepdim=True), min=1.0)
    X_mean = sum_vals / count_vals

    X_centered = (X - X_mean) * M
    X_var = torch.sum(X_centered ** 2, dim=0, keepdim=True) / count_vals
    X_std = torch.sqrt(X_var + 1e-8)

    return X_mean + torch.randn_like(X) * X_std * 0.1



# Training 
def train_cagi(trainX, trainM, params, device):
    """Train CAGI model with progressive alternating optimization.

    Args:
        trainX: Training data array (n_samples, n_features).
        trainM: Training mask array (n_samples, n_features).
        params: Parameter dictionary containing model hyperparameters.

    Returns:
        G: Trained generator.
        cluster_model: Fitted clustering model.
    """
    # Parameters
    mb_size = params['mb_size']
    p_hint = params['p_hint']
    n_clusters = params.get('n_clusters', 5)
    iterations = params.get('iterations', 5000)
    hidden_dim = params.get('hidden_dim', trainX.shape[1])
    cluster_method = params.get('cluster_method', 'kmeans')
    numerical_indices = params.get('numerical_indices', None)
    categorical_indices = params.get('categorical_indices', None)
    gamma = params.get('gamma', 0.5)
    alpha = params.get('alpha', 100.0)
    beta = params.get('beta', 1.0)
    patience = params.get('patience', 500)
    min_delta = params.get('min_delta', 1e-4)
    sinkhorn_freq = params.get('sinkhorn_freq', 5)
    cluster_update_freq = params.get('cluster_update_freq', 1000)

    X_data = torch.from_numpy(trainX).float().to(device)
    M_data = torch.from_numpy(trainM.astype(np.float32)).float().to(device)

    G = ImprovedGenerator(trainX.shape[1], hidden_dim, n_clusters).to(device)
    D = ImprovedDiscriminator(trainX.shape[1], hidden_dim).to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode='min', factor=0.5, patience=200)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(opt_D, mode='min', factor=0.5, patience=200)

    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.9, backend="tensorized")

    cluster_model = get_cluster_model(
        method=cluster_method, n_clusters=n_clusters, max_iter=100,
        numerical_indices=numerical_indices, categorical_indices=categorical_indices, gamma=gamma
    )
    cluster_model.fit(trainX, trainM)
    cluster_labels = cluster_model.labels_

    best_loss = float('inf')
    patience_counter = 0
    best_G_state = None

    for it in tqdm(range(iterations), desc="Training"):
        batch_idx = sample_batch_index(trainX.shape[0], mb_size)
        X_mb = X_data[batch_idx]
        M_mb = M_data[batch_idx]
        cluster_mb = cluster_onehot(batch_idx, cluster_labels, n_clusters).to(device)

        Z_mb = improved_noise_sampler(X_mb, M_mb, device)
        H_mb = M_mb * torch.from_numpy(binary_sampler(p_hint, mb_size, trainX.shape[1])).float().to(device)
        X_noised = M_mb * X_mb + (1 - M_mb) * Z_mb

        opt_D.zero_grad()
        with torch.no_grad():
            G_sample = G(X_noised, M_mb, cluster_mb)
        Hat_X = M_mb * X_mb + (1 - M_mb) * G_sample
        D_prob = D(Hat_X, H_mb)
        D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1 - M_mb) * torch.log(1 - D_prob + 1e-8))
        D_loss.backward()
        opt_D.step()

        opt_G.zero_grad()
        G_sample = G(X_noised, M_mb, cluster_mb)
        Hat_X = M_mb * X_mb + (1 - M_mb) * G_sample
        D_prob = D(Hat_X, H_mb)

        G_adv = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))

        MSE = torch.mean((M_mb * X_mb - M_mb * G_sample) ** 2) / torch.mean(M_mb)
        weighted_MSE = alpha * MSE

        if it % sinkhorn_freq == 0:
            perm = torch.randperm(Hat_X.size(0))
            Hat_X_shuffled = Hat_X[perm]
            half = Hat_X_shuffled.size(0) // 2
            sink = sinkhorn(Hat_X_shuffled[:half], Hat_X_shuffled[half:2 * half]) if half > 0 else torch.tensor(0.0).to(device)
        else:
            sink = torch.tensor(0.0).to(device)
        weighted_sinkhorn = beta * sink

        G_loss = G_adv + weighted_MSE + weighted_sinkhorn
        G_loss.backward()
        opt_G.step()

        if it % 100 == 0:
            scheduler_G.step(G_loss.item())
            scheduler_D.step(D_loss.item())

            current_loss = G_loss.item()
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
                best_G_state = G.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if it > 0 and it % cluster_update_freq == 0:
            with torch.no_grad():
                X_complete = []
                for i in range(0, len(trainX), mb_size):
                    end_idx = min(i + mb_size, len(trainX))
                    batch_idx_update = list(range(i, end_idx))
                    X_mb_u = X_data[batch_idx_update]
                    M_mb_u = M_data[batch_idx_update]
                    Z_mb_u = improved_noise_sampler(X_mb_u, M_mb_u, device)
                    X_noised_u = M_mb_u * X_mb_u + (1 - M_mb_u) * Z_mb_u
                    cluster_mb_u = cluster_onehot(batch_idx_update, cluster_labels, n_clusters).to(device)
                    G_sample_u = G(X_noised_u, M_mb_u, cluster_mb_u)
                    Hat_X_u = M_mb_u * X_mb_u + (1 - M_mb_u) * G_sample_u
                    X_complete.append(Hat_X_u.cpu().numpy())
                X_complete = np.vstack(X_complete)

            cluster_model = get_cluster_model(
                method=cluster_method, n_clusters=n_clusters, max_iter=100,
                numerical_indices=numerical_indices, categorical_indices=categorical_indices, gamma=gamma
            )
            cluster_model.fit(X_complete, trainM)
            cluster_labels = cluster_model.labels_

    if best_G_state is not None:
        G.load_state_dict(best_G_state)

    return G, cluster_model


# Test
def test_cagi(G, testX, testM, device, n_clusters, cluster_model=None):
    """Impute missing values in test data using trained CAGI model.
    Args:
        G: Trained generator.
        testX: Test data array.
        testM: Test mask array.
        device: PyTorch device.
        n_clusters: Number of clusters.
        cluster_model: Fitted clustering model for assigning test samples.
    Returns:
        X_imputed: Completed data array.
        mse: Reconstruction MSE.
    """
    G.eval()

    X_tensor = torch.from_numpy(testX).float().to(device)
    M_tensor = torch.from_numpy(testM.astype(np.float32)).float().to(device)

    Z_tensor = improved_noise_sampler(X_tensor, M_tensor, device)
    X_noised = M_tensor * X_tensor + (1 - M_tensor) * Z_tensor

    if cluster_model is not None:
        test_labels = cluster_model.predict(testX, testM)
        cluster_onehot_tensor = cluster_onehot(range(len(testX)), test_labels, n_clusters).to(device)
    else:
        cluster_onehot_tensor = torch.zeros((len(testX), n_clusters)).float().to(device)

    with torch.no_grad():
        G_sample = G(X_noised, M_tensor, cluster_onehot_tensor)

    X_imputed = M_tensor * X_tensor + (1 - M_tensor) * G_sample

    mse = torch.mean(
        ((1 - M_tensor) * X_tensor - (1 - M_tensor) * G_sample) ** 2
    ) / torch.mean(1 - M_tensor)

    return X_imputed.cpu().numpy(), mse.item()

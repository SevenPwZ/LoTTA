import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class MMDTestVisualizer:
    def __init__(self, sigmas=None, num_permutations=500, device='cuda'):
        self.sigmas = sigmas
        self.num_permutations = num_permutations
        self.device = device

    def _gaussian_kernel_matrix(self, X, sigma):
        """计算完整核矩阵 N×N"""
        XX = (X**2).sum(dim=1, keepdim=True)
        dists = XX - 2 * X @ X.t() + XX.t()
        return torch.exp(-dists / (2 * sigma**2))

    def _compute_mmd_from_kernel(self, K, idx_s, idx_t):
        """从预计算核矩阵 K 提取子矩阵计算 MMD"""
        n, m = len(idx_s), len(idx_t)
        Kxx = K[idx_s][:, idx_s]
        Kyy = K[idx_t][:, idx_t]
        Kxy = K[idx_s][:, idx_t]

        mmd = Kxx.sum() - Kxx.trace()
        mmd /= (n * (n - 1))
        mmd += Kyy.sum() - Kyy.trace()
        mmd /= (m * (m - 1))
        mmd -= 2 * Kxy.mean()
        return mmd

    def _auto_select_sigmas(self, X):
        """自动根据中位数距离生成多尺度 sigma"""
        with torch.no_grad():
            XX = (X**2).sum(dim=1, keepdim=True)
            dists = XX - 2 * X @ X.t() + XX.t()
            median_dist = torch.median(dists[dists > 0]) ** 0.5
            return [median_dist/2, median_dist, median_dist*2, median_dist*4]

    def compute_p_value(self, X_s, X_t):
        X_s = X_s.data.to(self.device)
        X_t = X_t.dataset.data[X_t.indices]
        X_t = X_t.data.to(self.device)

        # 转换为 tensor 并标准化
        X_s = torch.tensor(X_s, dtype=torch.float32, device=self.device)
        X_t = torch.tensor(X_t, dtype=torch.float32, device=self.device)
        X_all = torch.cat([X_s, X_t], dim=0)
        X_all = (X_all - X_all.mean(dim=0)) / (X_all.std(dim=0) + 1e-8)

        n_s = X_s.shape[0]
        idx_s = torch.arange(0, n_s, device=self.device)
        idx_t = torch.arange(n_s, X_all.shape[0], device=self.device)

        # 自动选择 sigma
        sigmas = self.sigmas or self._auto_select_sigmas(X_all)

        # 聚合多带宽结果
        def aggregate_mmd(idx_s, idx_t, K_list):
            mmd_vals = [self._compute_mmd_from_kernel(K, idx_s, idx_t) for K in K_list]
            return torch.stack(mmd_vals).max()

        # 预计算所有 sigma 的核矩阵
        K_list = [self._gaussian_kernel_matrix(X_all, sigma) for sigma in sigmas]

        # 真实 MMD 统计量
        stat_obs = aggregate_mmd(idx_s, idx_t, K_list)

        # 置换检验
        perm_stats = []
        for _ in range(self.num_permutations):
            idx_perm = torch.randperm(X_all.shape[0], device=self.device)
            idx_s_perm = idx_perm[:n_s]
            idx_t_perm = idx_perm[n_s:]
            stat_perm = aggregate_mmd(idx_s_perm, idx_t_perm, K_list)
            perm_stats.append(stat_perm)

        perm_stats = torch.stack(perm_stats)
        p_value = ((perm_stats >= stat_obs).sum().float() + 1) / (self.num_permutations + 1)

        return p_value.item()

    def plot_kde(self, X_s, X_t, feature_idx=0):
        """绘制单维特征核密度对比图"""
        X_s = np.array(X_s)
        X_t = np.array(X_t)
        xs = X_s[:, feature_idx]
        xt = X_t[:, feature_idx]

        kde_s = gaussian_kde(xs)
        kde_t = gaussian_kde(xt)

        x_min, x_max = min(xs.min(), xt.min()), max(xs.max(), xt.max())
        grid = np.linspace(x_min, x_max, 200)

        plt.figure(figsize=(6, 4))
        plt.plot(grid, kde_s(grid), label='Train (Normal)', color='blue')
        plt.plot(grid, kde_t(grid), label='Test (Normal Selected)', color='orange')
        plt.fill_between(grid, kde_s(grid), alpha=0.3, color='blue')
        plt.fill_between(grid, kde_t(grid), alpha=0.3, color='orange')
        plt.legend()
        plt.title(f"KDE Comparison on Feature {feature_idx}")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")
        plt.show()
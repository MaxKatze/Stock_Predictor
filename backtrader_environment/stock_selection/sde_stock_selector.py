"""SDE-based stock selection using macroeconomic factors and Euler-Maruyama simulation."""

import numpy as np
import pandas as pd
from itertools import combinations
import warnings
import matplotlib.pyplot as plt
import os


class SDEStockSelector:
    """Select stocks from a universe using Stochastic Differential Equations.

    Algorithm:
    1. Discretize daily movements to {+1, 0, -1}
    2. For each stock, find dominant macro factor pair via OLS regression
    3. Group stocks by dominant factor pair
    4. Form simulation groups (4 stocks + 2 factors)
    5. Simulate via Euler-Maruyama, select winner per group
    6. Repeat N times, rank by average placement
    """

    def __init__(self, num_simulations=100, group_size=4, num_factors=2, num_selected=10,
                 output_dir=None):
        self.num_simulations = num_simulations
        self.group_size = group_size
        self.num_factors = num_factors
        self.num_selected = num_selected
        self.output_dir = output_dir

        self.dominant_pairs = {}
        self.beta_sums = {}
        self.avg_roi = {}
        self.example_simulation = None

    def select(self, stock_data: dict[str, pd.Series],
               factor_data: dict[str, pd.Series],
               simulation_days: int = 378) -> list[str]:
        """Run the full stock selection process.

        Args:
            stock_data: dict mapping ticker -> Series of close prices (training period)
            factor_data: dict mapping factor_key -> Series of close prices (training period)
            simulation_days: number of trading days to simulate (test period length)

        Returns:
            List of selected ticker symbols (sorted by rank)
        """
        tickers = list(stock_data.keys())
        factor_keys = list(factor_data.keys())

        stock_directions = self._discretize_all(stock_data)
        factor_directions = self._discretize_all(factor_data)

        dominant_pairs, beta_sums = self._find_dominant_pairs(
            stock_directions, factor_directions, tickers, factor_keys
        )
        self.dominant_pairs = dominant_pairs
        self.beta_sums = beta_sums

        pools = self._group_by_factor_pair(tickers, dominant_pairs, factor_keys)

        rankings = {t: [] for t in tickers}
        roi_accumulator = {t: [] for t in tickers}
        example_saved = False

        for iteration in range(self.num_simulations):
            groups = self._form_simulation_groups(pools)

            for group_tickers, factor_pair in groups:
                result = self._simulate_group(
                    group_tickers, factor_pair,
                    stock_directions, factor_directions,
                    stock_data, simulation_days,
                    save_example=(not example_saved and iteration == 0)
                )
                if result is None:
                    continue

                winner, group_rois, simulation_history = result

                if simulation_history is not None and not example_saved:
                    self.example_simulation = {
                        "tickers": group_tickers,
                        "factors": factor_pair,
                        "history": simulation_history,
                    }
                    self._save_example_plot(group_tickers, factor_pair, simulation_history)
                    example_saved = True

                for t in group_tickers:
                    roi_accumulator[t].append(group_rois[t])

                for i, t in enumerate(sorted(
                    group_tickers,
                    key=lambda x: x == winner,
                    reverse=True
                )):
                    rankings[t].append(i)

        for t in tickers:
            if roi_accumulator[t]:
                self.avg_roi[t] = np.mean(roi_accumulator[t])
            else:
                self.avg_roi[t] = float("-inf")

        avg_rankings = {}
        for t in tickers:
            if rankings[t]:
                avg_rankings[t] = np.mean(rankings[t])
            else:
                avg_rankings[t] = float("inf")

        sorted_tickers = sorted(avg_rankings, key=avg_rankings.get)
        return sorted_tickers[:self.num_selected]

    def get_detailed_results(self):
        """Return detailed results for reporting."""
        return {
            "dominant_pairs": self.dominant_pairs,
            "beta_sums": self.beta_sums,
            "avg_roi": self.avg_roi,
            "example_simulation": self.example_simulation,
        }

    def _discretize_all(self, data: dict[str, pd.Series]) -> dict[str, np.ndarray]:
        """Discretize price series to {+1, 0, -1} based on daily direction."""
        result = {}
        for key, series in data.items():
            prices = series.values
            diffs = np.diff(prices)
            directions = np.sign(diffs).astype(int)
            result[key] = directions
        return result

    def _find_dominant_pairs(self, stock_dirs, factor_dirs, tickers, factor_keys):
        """For each stock, find the 2 factors with highest |beta| sum."""
        T = min(len(next(iter(stock_dirs.values()))),
                min(len(v) for v in factor_dirs.values()))

        F_matrix = np.column_stack([factor_dirs[k][:T] for k in factor_keys])
        X = np.column_stack([np.ones(T), F_matrix])

        dominant_pairs = {}
        beta_sums = {}
        for ticker in tickers:
            y = stock_dirs[ticker][:T].astype(float)
            try:
                theta = np.linalg.lstsq(X, y, rcond=None)[0]
                betas = np.abs(theta[1:])
            except Exception:
                betas = np.zeros(len(factor_keys))

            factor_indices = list(range(len(factor_keys)))
            best_pair = None
            best_sum = -1
            for i, j in combinations(factor_indices, 2):
                beta_sum = betas[i] + betas[j]
                if beta_sum > best_sum:
                    best_sum = beta_sum
                    best_pair = (factor_keys[i], factor_keys[j])

            dominant_pairs[ticker] = best_pair
            beta_sums[ticker] = best_sum

        return dominant_pairs, beta_sums

    def _group_by_factor_pair(self, tickers, dominant_pairs, factor_keys):
        """Group stocks into pools by their dominant factor pair."""
        pools = {}
        for ticker in tickers:
            pair = dominant_pairs[ticker]
            pair_key = tuple(sorted(pair))
            if pair_key not in pools:
                pools[pair_key] = []
            pools[pair_key].append(ticker)
        return pools

    def _save_example_plot(self, group_tickers, factor_pair, history):
        """Save example simulation plot to file."""
        if self.output_dir is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        days = np.arange(history["stocks"].shape[0])

        for i, ticker in enumerate(group_tickers):
            normalized = history["stocks"][:, i] * 100
            ax.plot(days, normalized, label=ticker, linewidth=1.5)

        for i, factor in enumerate(factor_pair):
            normalized = history["factors"][:, i] * 100
            ax.plot(days, normalized, label=factor, linewidth=1.5, linestyle='--')

        ax.set_xlabel("Simulationstage")
        ax.set_ylabel("Normalisierter Wert (Start = 100)")
        ax.set_title("Euler-Maruyama Simulation einer Gruppe")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "example_simulation.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _form_simulation_groups(self, pools):
        """Randomly form simulation groups of `group_size` stocks from each pool."""
        groups = []
        for factor_pair, pool_tickers in pools.items():
            np.random.shuffle(pool_tickers)
            for i in range(0, len(pool_tickers), self.group_size):
                group = pool_tickers[i:i + self.group_size]
                if len(group) >= 2:
                    groups.append((group, factor_pair))
        return groups

    def _simulate_group(self, group_tickers, factor_pair,
                        stock_dirs, factor_dirs,
                        stock_data, simulation_days,
                        save_example=False):
        """Simulate one group via Euler-Maruyama and return the winner ticker, ROIs, and optionally history."""
        g = len(group_tickers)
        f = len(factor_pair)
        dim = g + f

        T = min(len(stock_dirs[group_tickers[0]]),
                min(len(factor_dirs[k]) for k in factor_pair))

        directions_matrix = np.column_stack(
            [stock_dirs[t][:T] for t in group_tickers] +
            [factor_dirs[k][:T] for k in factor_pair]
        )

        num_combinations = 3 ** dim
        combo_counts = {}

        for t_idx in range(T):
            combo = tuple(directions_matrix[t_idx])
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

        all_combos = _generate_all_combinations(dim)

        probabilities = {}
        for combo in all_combos:
            n_c = combo_counts.get(combo, 0)
            probabilities[combo] = (n_c + 1) / (T + num_combinations)

        mu = np.zeros(dim)
        for combo in all_combos:
            c = np.array(combo, dtype=float)
            mu += probabilities[combo] * c

        sigma = np.zeros((dim, dim))
        for combo in all_combos:
            c = np.array(combo, dtype=float).reshape(-1, 1)
            sigma += probabilities[combo] * (c @ c.T)

        sigma += np.eye(dim) * 1e-8

        try:
            B = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            eigenvalues = np.linalg.eigvalsh(sigma)
            sigma += np.eye(dim) * (abs(min(eigenvalues)) + 1e-6)
            B = np.linalg.cholesky(sigma)

        dt = 1.0
        sqrt_dt = np.sqrt(dt)

        start_prices = np.array([
            stock_data[t].iloc[-1] for t in group_tickers
        ])

        S = np.ones(g)
        F = np.ones(f)

        history = None
        if save_example:
            history = {"stocks": np.zeros((simulation_days + 1, g)),
                       "factors": np.zeros((simulation_days + 1, f))}
            history["stocks"][0] = S.copy()
            history["factors"][0] = F.copy()

        for day in range(simulation_days):
            Z = np.random.randn(dim)
            dW = sqrt_dt * Z
            noise = B @ dW
            S = S + mu[:g] * dt + noise[:g]
            F = F + mu[g:] * dt + noise[g:]

            if save_example:
                history["stocks"][day + 1] = S.copy()
                history["factors"][day + 1] = F.copy()

        roi = (S - 1.0) / 1.0
        winner_idx = np.argmax(roi)

        group_rois = {t: roi[i] for i, t in enumerate(group_tickers)}

        return group_tickers[winner_idx], group_rois, history if save_example else None


def _generate_all_combinations(dim):
    """Generate all {-1, 0, 1}^dim combinations."""
    if dim == 0:
        return [()]
    sub = _generate_all_combinations(dim - 1)
    result = []
    for val in [-1, 0, 1]:
        for s in sub:
            result.append((val,) + s)
    return result

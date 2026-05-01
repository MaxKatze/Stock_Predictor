"""SDE-based stock selection using macroeconomic factors and Euler-Maruyama simulation."""

import numpy as np
import pandas as pd
from itertools import combinations
import warnings


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

    def __init__(self, num_simulations=100, group_size=4, num_factors=2, num_selected=10):
        self.num_simulations = num_simulations
        self.group_size = group_size
        self.num_factors = num_factors
        self.num_selected = num_selected

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

        dominant_pairs = self._find_dominant_pairs(
            stock_directions, factor_directions, tickers, factor_keys
        )

        pools = self._group_by_factor_pair(tickers, dominant_pairs, factor_keys)

        rankings = {t: [] for t in tickers}

        for iteration in range(self.num_simulations):
            groups = self._form_simulation_groups(pools)

            for group_tickers, factor_pair in groups:
                winner = self._simulate_group(
                    group_tickers, factor_pair,
                    stock_directions, factor_directions,
                    stock_data, simulation_days
                )
                if winner is not None:
                    for i, t in enumerate(sorted(
                        group_tickers,
                        key=lambda x: x == winner,
                        reverse=True
                    )):
                        rankings[t].append(i)

        avg_rankings = {}
        for t in tickers:
            if rankings[t]:
                avg_rankings[t] = np.mean(rankings[t])
            else:
                avg_rankings[t] = float("inf")

        sorted_tickers = sorted(avg_rankings, key=avg_rankings.get)
        return sorted_tickers[:self.num_selected]

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

        return dominant_pairs

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
                        stock_data, simulation_days):
        """Simulate one group via Euler-Maruyama and return the winner ticker."""
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
        S = start_prices.copy()

        for _ in range(simulation_days):
            Z = np.random.randn(dim)
            dW = sqrt_dt * Z
            noise = B @ dW
            S = S + mu[:g] * dt + noise[:g]

        roi = (S - start_prices) / start_prices
        winner_idx = np.argmax(roi)
        return group_tickers[winner_idx]


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

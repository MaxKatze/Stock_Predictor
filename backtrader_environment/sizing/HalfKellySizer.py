"""Half-Kelly Criterion Position Manager as specified in Konzeption Section 3.4.3.

The Kelly criterion maximizes long-term capital growth rate. Half-Kelly is used
to reduce volatility and sensitivity to estimation errors.

Formula:
    f_t = 0.5 * μ_t / σ²_t

Where:
    μ_t = mean return over last w_K days (rolling window)
    σ²_t = variance of returns over last w_K days
    w_K = 60 trading days (default)

The fraction f_t is clipped to [0, 1] since no short-selling or leverage.

IMPORTANT (from Konzeption):
"Ist also zum Zeitpunkt t Signal_t = Long, wird die Position so angepasst,
dass sie dem Anteil f_t vom Gesamtportfoliowert entspricht.
Dies kann auch einer Reduktion der Position entsprechen."

This means when Signal=Long, position is ADJUSTED to f_t * portfolio_value,
which may involve SELLING if current position is too large.
"""

import numpy as np


class HalfKellyPositionManager:
    """Calculates Half-Kelly position fraction and target position size.

    This is not a Backtrader Sizer - it's a utility class used by the strategy
    to determine target position sizes, including reductions.

    Parameters:
        kelly_window (int): Rolling window for μ and σ² estimation (default: 60)
        min_observations (int): Minimum observations before applying Kelly (default: 30)
    """

    def __init__(self, kelly_window=60, min_observations=30):
        self.kelly_window = kelly_window
        self.min_observations = min_observations
        self._returns_buffer = {}

    def compute_kelly_fraction(self, data) -> float:
        """Compute the Half-Kelly fraction f_t for the given data.

        Args:
            data: Backtrader data feed

        Returns:
            f_t in [0, 1] - the fraction of portfolio to invest
        """
        if data not in self._returns_buffer:
            self._returns_buffer[data] = []

        closes = list(data.close.get(size=min(len(data), self.kelly_window + 1)))

        if len(closes) < 2:
            return 0.5

        returns = []
        for i in range(1, len(closes)):
            r = np.log(closes[i] / closes[i - 1])
            returns.append(r)

        self._returns_buffer[data] = returns[-self.kelly_window:]

        if len(self._returns_buffer[data]) < self.min_observations:
            return 0.5

        returns_arr = np.array(self._returns_buffer[data])
        mu = np.mean(returns_arr)
        sigma_sq = np.var(returns_arr, ddof=1)

        if sigma_sq < 1e-10:
            return 0.5

        f_t = 0.5 * mu / sigma_sq
        return float(np.clip(f_t, 0.0, 1.0))

    def compute_target_position(self, data, portfolio_value: float, current_price: float) -> int:
        """Compute target position size in shares.

        Args:
            data: Backtrader data feed
            portfolio_value: Total portfolio value
            current_price: Current share price

        Returns:
            Target number of shares (may be less than current position)
        """
        f_t = self.compute_kelly_fraction(data)
        target_value = portfolio_value * f_t
        target_shares = int(target_value / current_price)
        return max(0, target_shares)

    def compute_order_size(self, data, portfolio_value: float, current_position: int,
                          current_price: float, available_cash: float) -> int:
        """Compute order size to reach target position.

        Args:
            data: Backtrader data feed
            portfolio_value: Total portfolio value
            current_position: Current number of shares held
            current_price: Current share price
            available_cash: Available cash for buying

        Returns:
            Order size: positive = buy, negative = sell, 0 = no action
        """
        target_shares = self.compute_target_position(data, portfolio_value, current_price)
        order_size = target_shares - current_position

        if order_size > 0:
            max_buyable = int((available_cash * 0.99) / current_price)
            order_size = min(order_size, max_buyable)

        return order_size

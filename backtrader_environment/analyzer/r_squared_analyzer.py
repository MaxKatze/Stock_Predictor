from .general_analyzer import GeneralAnalyzer


class RSquaredAnalyzer(GeneralAnalyzer):
    """
    R² Analyzer that compares PREDICTED RETURNS with ACTUAL RETURNS.

    This is more meaningful than comparing predicted prices with actual prices,
    because prices are highly autocorrelated (today's price ≈ yesterday's price).
    A naive model predicting "no change" would get R² ≈ 0.99 on prices.

    By comparing returns, we measure how well the model predicts the
    DIRECTION and MAGNITUDE of price movements.
    """

    def __init__(self):
        super().__init__()
        self.r2 = {}
        for d in self.strategy.datas:
            self.r2[d._name] = 0

    def stop(self):
        for d in self.strategy.datas:
            # Check if strategy tracks returns (LSTM strategy)
            if hasattr(self.strategy, 'predicted_returns') and hasattr(self.strategy, 'actual_returns'):
                self._calculate_r2_from_returns(d)
            else:
                # Fallback to price-based R² for other strategies
                self._calculate_r2_from_prices(d)

    def _calculate_r2_from_returns(self, d):
        """Calculate R² comparing predicted returns with actual returns."""
        predicted = self.strategy.predicted_returns[d]
        actual = self.strategy.actual_returns[d]

        # Align: predicted_returns[i] is the prediction made on day i for day i+1
        # actual_returns[i] is the return from day i to day i+1
        # So we need to compare predicted_returns[:-1] with actual_returns[1:]
        # But actual_returns starts from day 2 (needs 2 prices), so:
        # predicted_returns[i] corresponds to actual_returns[i] (both for return from day i to i+1)

        # Filter out None predictions and align with actual returns
        # predicted_returns has one more entry (last prediction has no actual yet)
        pairs = []
        for i in range(min(len(predicted), len(actual))):
            if predicted[i] is not None:
                pairs.append((predicted[i], actual[i]))

        if len(pairs) < 3:
            self.r2[d._name] = -1
            return

        pred_returns = [p for p, _ in pairs]
        act_returns = [a for _, a in pairs]

        # Calculate R²
        mean_actual = sum(act_returns) / len(act_returns)
        ss_tot = sum((a - mean_actual) ** 2 for a in act_returns)

        if ss_tot == 0:
            self.r2[d._name] = -1
            return

        ss_res = sum((a - p) ** 2 for p, a in pairs)
        self.r2[d._name] = 1 - (ss_res / ss_tot)

    def _calculate_r2_from_prices(self, d):
        """Fallback: Calculate R² comparing predicted prices with actual prices."""
        if len(self.strategy.prices[d]) != len(self.strategy.predictions[d]):
            self.r2[d._name] = -1
            return

        pairs = [
            (price, prediction)
            for price, prediction in zip(self.strategy.prices[d], self.strategy.predictions[d])
            if prediction is not None
        ]
        if len(pairs) < 3:
            self.r2[d._name] = -1
            return

        observed_prices = [price for price, _ in pairs]
        mean_price = sum(observed_prices) / len(observed_prices)
        ss_tot = sum((price - mean_price) ** 2 for price, _ in pairs)
        if ss_tot == 0:
            self.r2[d._name] = -1
            return

        ss_res = sum((price - prediction) ** 2 for price, prediction in pairs)
        self.r2[d._name] = 1 - (ss_res / ss_tot)

    def get_analysis(self):
        super_analysis = super().get_analysis()
        analysis = {}
        analysis["general"] = super_analysis
        analysis["r_squared_analysis"] = self.r2
        return analysis

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
from numpy.linalg import inv
import warnings

warnings.filterwarnings('ignore')

from data_loader import DataLoader
from factor_selection import IterativeFactorSelection


class GlobalFactorAnalysis:

    def __init__(self, weighting_scheme='VW_cap', start_date='1993-08-01', end_date='2021-12-31'):
        self.weighting_scheme = weighting_scheme
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}

    def load_data(self, region='world'):
        data_loader = DataLoader(self.weighting_scheme, self.start_date, self.end_date)
        factors_df, market_return = data_loader.load_factor_data(region)
        return factors_df.fillna(0), market_return.fillna(0)

    def select_global_factors(self, max_factors=30):
        self.global_factors, self.global_market = self.load_data('world')
        selector = IterativeFactorSelection(self.global_factors, self.global_market)
        global_results = selector.select_factors_t_std(max_factors=max_factors)
        self.selected_factors = global_results['factor'].tolist()
        return global_results

    def analyze_regions(self):
        # Load all regions data
        self.us_factors, self.us_market = self.load_data('US')
        self.exus_factors, self.exus_market = self.load_data('ex US')

        # Initialize results storage
        regions = ['World', 'US', 'World_ex_US']
        metrics = ['GRS', 'p_value', 't2', 't3']
        self.results = {region: {metric: [] for metric in metrics} for region in regions}

        # Analyze for each number of factors
        for n in range(1, len(self.selected_factors) + 1):
            factors = self.selected_factors[:n]

            for region, factors_df, market_return in [
                ('World', self.global_factors, self.global_market),
                ('US', self.us_factors, self.us_market),
                ('World_ex_US', self.exus_factors, self.exus_market)
            ]:
                stats = self._compute_stats(factors_df, market_return, factors)
                self.results[region]['GRS'].append(stats['grs'])
                self.results[region]['p_value'].append(stats['p_value'])
                self.results[region]['t2'].append(stats['t2'])
                self.results[region]['t3'].append(stats['t3'])

        return self._create_results_table()

    def _compute_stats(self, factors_df, market_return, selected_factors):
        X = pd.concat([market_return.to_frame('market')] +
                      [factors_df[f] for f in selected_factors], axis=1)

        alphas, residuals, t_stats = [], [], []
        remaining_factors = [f for f in factors_df.columns if f not in selected_factors and f != 'market_equity']

        for factor in remaining_factors:
            y = factors_df[factor]
            valid_idx = ~(y.isna() | X.isna().any(axis=1))

            if valid_idx.sum() >= 50:
                X_const = sm.add_constant(X[valid_idx])
                model = sm.OLS(y[valid_idx], X_const)
                res = model.fit()
                alphas.append(res.params[0])
                residuals.append(res.resid)
                t_stats.append(res.tvalues[0])

        stats = {
            't2': sum(abs(t) > 1.96 for t in t_stats),
            't3': sum(abs(t) > 3.0 for t in t_stats),
            'grs': np.nan,
            'p_value': np.nan
        }

        if len(alphas) > 0:
            stats.update(self._calculate_grs(alphas, residuals, X))

        return stats

    def _calculate_grs(self, alphas, residuals, factors):
        try:
            T, N = len(residuals[0]), len(alphas)
            K = factors.shape[1] if factors.ndim > 1 else 1

            alphas = np.array(alphas).reshape(-1, 1)
            residuals = np.column_stack(residuals)
            factors = factors.values if isinstance(factors, pd.DataFrame) else factors

            Sigma = np.cov(residuals.T)
            Omega = np.cov(factors.T) if factors.ndim > 1 else np.array([[np.var(factors)]])

            f_bar = np.mean(factors, axis=0).reshape(-1, 1) if factors.ndim > 1 else np.array([[np.mean(factors)]])

            Sh2_alpha = float(alphas.T @ inv(Sigma) @ alphas)
            Sh2_f = float(f_bar.T @ inv(Omega) @ f_bar)

            grs = ((T - N - K) / N) * (Sh2_alpha / (1 + Sh2_f))
            p_value = 1 - f.cdf(grs, N, T - N - K)

            return {'grs': grs, 'p_value': p_value}
        except:
            return {'grs': np.nan, 'p_value': np.nan}

    def _create_results_table(self):
        table = []

        for i, factor in enumerate(self.selected_factors, 1):
            row = {
                'No.': i,
                'Factor': factor,
                'World_GRS': self._format(self.results['World']['GRS'][i - 1]),
                'World_p(GRS)': self._format(self.results['World']['p_value'][i - 1], is_pvalue=True),
                'World_t>2': self.results['World']['t2'][i - 1],
                'World_t>3': self.results['World']['t3'][i - 1],
                'US_GRS': self._format(self.results['US']['GRS'][i - 1]),
                'US_p(GRS)': self._format(self.results['US']['p_value'][i - 1], is_pvalue=True),
                'US_t>2': self.results['US']['t2'][i - 1],
                'US_t>3': self.results['US']['t3'][i - 1],
                'World_ex_US_GRS': self._format(self.results['World_ex_US']['GRS'][i - 1]),
                'World_ex_US_p(GRS)': self._format(self.results['World_ex_US']['p_value'][i - 1], is_pvalue=True),
                'World_ex_US_t>2': self.results['World_ex_US']['t2'][i - 1],
                'World_ex_US_t>3': self.results['World_ex_US']['t3'][i - 1]
            }
            table.append(row)

        return pd.DataFrame(table)

    def _format(self, value, is_pvalue=False):
        if pd.isna(value):
            return "-"
        return round(value, 3) if is_pvalue else round(value, 2)


def run_analysis():
    analyzer = GlobalFactorAnalysis()
    print("Sélection des facteurs globaux...")
    analyzer.select_global_factors()
    print("Analyse par région...")
    results_df = analyzer.analyze_regions()
    return results_df


if __name__ == "__main__":
    results = run_analysis()
    print("\nRésultats finaux :")
    print(results.to_string())
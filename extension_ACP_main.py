import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from extension_ACP import PCAPurification

class FactorSelectionAdapter:
    def __init__(self, factors_df, market_return, max_factors=10):
        self.factors_df = factors_df
        self.market_return = market_return
        self.max_factors = max_factors
        self.results = []

    def run_regression(self, y, X):
        import statsmodels.api as sm
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const, missing='drop')
        res = model.fit()
        return {
            'alpha': res.params[0],
            't_stat': res.tvalues[0],
            'p_value': res.pvalues[0],
            'residuals': res.resid,
            'r_squared': res.rsquared
        }

    def select_factors(self):
        available_factors = list(self.factors_df.columns)
        selected_factors = []
        results = []

        for iteration in range(self.max_factors):
            print(f"\n--- Itération {iteration + 1} ---")

            if iteration == 0:
                X_base = self.market_return.to_frame('market')
            else:
                X_base = pd.concat([self.market_return.to_frame('market')] +
                                   [self.factors_df[f] for f in selected_factors], axis=1)

            factor_results = {}
            best_factor = None
            best_t_stat = 0

            for factor in available_factors:
                y = self.factors_df[factor]
                valid_idx = ~(y.isna() | X_base.isna().any(axis=1))
                if valid_idx.sum() < 30:
                    continue

                reg_results = self.run_regression(y[valid_idx], X_base[valid_idx])
                factor_results[factor] = {
                    'alpha': reg_results['alpha'],
                    't_stat': reg_results['t_stat'],
                    'r_squared': reg_results['r_squared']
                }

                if abs(reg_results['t_stat']) > abs(best_t_stat):
                    best_factor = factor
                    best_t_stat = reg_results['t_stat']

            if best_factor is None:
                print("Aucun facteur sélectionné - arrêt")
                break

            selected_factors.append(best_factor)
            available_factors.remove(best_factor)
            results.append({
                'iteration': iteration + 1,
                'factor': best_factor,
                't_stat': best_t_stat,
                'factors_in_model': len(selected_factors),
                'r_squared': factor_results[best_factor]['r_squared']
            })

            print(f"Facteur sélectionné: {best_factor} (t-stat: {best_t_stat:.3f})")

            if abs(best_t_stat) < 2.0:
                print(f"Arrêt: t-stat ({best_t_stat:.3f}) inférieur à 2.0")
                break

        return pd.DataFrame(results)



def main_pca_extension(
    weighting_schemes=['VW_cap'], 
    start_date='1993-08-01', 
    end_date='2021-12-31',
    region='world',
    n_components_range=[5, 10, 15],
    visualization_dir='results_pca'
):
    os.makedirs(visualization_dir, exist_ok=True)
    results_dict = {}

    for scheme in weighting_schemes:
        print(f"\n{'='*80}")
        print(f"ANALYSE PCA POUR LE SCHÉMA DE PONDÉRATION: {scheme}")
        print(f"{'='*80}")

        data_loader = DataLoader(scheme, start_date, end_date)
        factors_df, market_return = data_loader.load_factor_data(region=region)

        diagnostic = data_loader.diagnostic_check(factors_df, market_return)
        if diagnostic['likely_percentage']:
            print("Conversion des données de pourcentage en décimal")
            factors_df = factors_df / 100
            market_return = market_return / 100

        scheme_dir = os.path.join(visualization_dir, scheme)
        os.makedirs(scheme_dir, exist_ok=True)

        for n_components in n_components_range:
            print(f"\nAnalyse avec {n_components} composantes principales:")

            pca_extension = PCAPurification(factors_df, market_return)
            pca_extension.preprocess_data()
            fa_results = pca_extension.run_factor_analysis_varimax(n_components=n_components)
            purified_factors = pca_extension.purify_factors(n_components=n_components)

            comp_dir = os.path.join(scheme_dir, f"components_{n_components}")
            os.makedirs(comp_dir, exist_ok=True)

            fig_scree = pca_extension.plot_scree(n_components=n_components)
            fig_scree.savefig(os.path.join(comp_dir, 'scree_plot.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_scree)

            fig_loadings = pca_extension.plot_factor_loadings(n_components=min(n_components, 5))
            fig_loadings.savefig(os.path.join(comp_dir, 'factor_loadings.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_loadings)

            fig_dendro = pca_extension.plot_factor_dendrogram()
            fig_dendro.savefig(os.path.join(comp_dir, 'factor_dendrogram.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_dendro)

            fig_comparison = pca_extension.compare_original_vs_purified(factors_to_show=10, n_components=n_components)
            fig_comparison.savefig(os.path.join(comp_dir, 'original_vs_purified.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_comparison)

            results_df, fig_eval = pca_extension.evaluate_purified_factors(market_return=market_return, n_components=n_components)
            fig_eval.savefig(os.path.join(comp_dir, 'evaluation.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_eval)

            fig_heatmap_orig = pca_extension.plot_correlation_heatmap(n_top_factors=30, purified=False)
            fig_heatmap_orig.savefig(os.path.join(comp_dir, 'heatmap_original.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_heatmap_orig)

            fig_heatmap_purif = pca_extension.plot_correlation_heatmap(n_top_factors=30, purified=True)
            fig_heatmap_purif.savefig(os.path.join(comp_dir, 'heatmap_purified.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_heatmap_purif)

            # Utilisation d'une clé plus consistante en format pour éviter les problèmes de parsing
            result_key = f"{scheme}_components_{n_components}"
            
            results_dict[result_key] = {
                'pca_results': fa_results,
                'purified_factors': purified_factors,
                'evaluation': results_df
            }

            print(f"Analyses et visualisations sauvegardées dans: {comp_dir}")

            print("\nSélection itérative de facteurs sur les facteurs purifiés:")
            try:
                temp_file = f"{scheme}_purified_{n_components}.csv"
                purified_factors.to_csv(temp_file)

                purified_selector = FactorSelectionAdapter(
                    purified_factors, 
                    market_return,
                    max_factors=10
                )

                purified_results = purified_selector.select_factors()
                purified_results.to_csv(os.path.join(comp_dir, 'purified_factor_selection.csv'))

                print("Top 5 facteurs purifiés sélectionnés:")
                print(purified_results.head(5))

                results_dict[result_key]['selection_results'] = purified_results

            except Exception as e:
                print(f"Erreur lors de la sélection des facteurs purifiés: {e}")

    return results_dict
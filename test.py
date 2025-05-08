import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
from numpy.linalg import inv
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
class DataLoader:
 
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        
    def load_factor_data(self, weighting_scheme: str = 'CW') -> Tuple[pd.DataFrame, pd.Series]:

        # Exemple avec données simulées (remplacer par vos vraies données)
        np.random.seed(42)
        n_periods = 600
        n_factors = 153  # Comme dans l'article
        
        # Dates
        dates = pd.date_range('1971-11-01', periods=n_periods, freq='M')
        
        # Rendements du marché
        market_return = pd.Series(
            np.random.normal(0.01, 0.04, n_periods),
            index=dates,
            name='market'
        )
        
        # Créer des facteurs avec différentes caractéristiques
        factors_data = {}
        
        # Facteurs avec de vrais alphas (environ 15 selon l'article)
        for i in range(15):
            alpha = np.random.normal(0.003, 0.001) * (15 - i) / 15
            beta = np.random.uniform(0.5, 1.5)
            
            if weighting_scheme == 'EW':
                # Les facteurs EW ont des alphas plus forts
                alpha *= 1.5
                noise = np.random.normal(0, 0.05, n_periods)
            elif weighting_scheme == 'VW':
                noise = np.random.normal(0, 0.04, n_periods)
            else:  # CW
                noise = np.random.normal(0, 0.035, n_periods)
            
            factor_return = alpha + beta * market_return + noise
            factors_data[f'factor_{i+1}'] = factor_return
        
        # Facteurs sans alpha significatif
        for i in range(15, n_factors):
            beta = np.random.uniform(0.3, 1.2)
            if weighting_scheme == 'EW':
                noise = np.random.normal(0, 0.06, n_periods)
            else:
                noise = np.random.normal(0, 0.04, n_periods)
            
            factor_return = beta * market_return + noise
            factors_data[f'factor_{i+1}'] = factor_return
        
        factors_df = pd.DataFrame(factors_data, index=dates)
        
        # Nettoyer les données (supprimer les NaN)
        factors_df = factors_df.dropna()
        market_return = market_return[factors_df.index]
        
        return factors_df, market_return


# 2. TEST GRS
class GRSTest:
    
    @staticmethod
    def calculate_grs(alphas: np.ndarray, residuals: np.ndarray, 
                     factors: np.ndarray) -> Tuple[float, float]:

        T, N = residuals.shape
        K = factors.shape[1]
        
        # S'assurer que alphas est un vecteur colonne
        alphas = np.array(alphas).reshape(-1, 1)
        
        # Matrices de covariance
        Sigma = np.cov(residuals.T, bias=False)
        Omega = np.cov(factors.T, bias=False)
        
        # Moyenne des facteurs
        f_bar = np.mean(factors, axis=0).reshape(-1, 1)
        
        # Ratios de Sharpe au carré
        Sh2_alpha = float(alphas.T @ inv(Sigma) @ alphas)
        Sh2_f = float(f_bar.T @ inv(Omega) @ f_bar)
        
        # Statistique GRS
        grs_stat = ((T - N - K) / N) * ((T - K - 1) / (T - K - 1)) * (Sh2_alpha / (1 + Sh2_f))
        
        # p-value
        p_value = 1 - f.cdf(grs_stat, N, T - N - K)
        
        return grs_stat, p_value


# 3. SÉLECTION ITÉRATIVE NESTED
class IterativeFactorSelection:
    
    def __init__(self, factors_df: pd.DataFrame, market_return: pd.Series,
                 significance_threshold: float = 3.0):
        self.factors_df = factors_df
        self.market_return = market_return
        self.significance_threshold = significance_threshold
        self.results = []
        
    def run_regression(self, y: pd.Series, X: pd.DataFrame) -> Dict:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const, missing='drop')
        res = model.fit()
        
        return {
            'alpha': res.params[0],
            't_stat': res.tvalues[0],
            'p_value': res.pvalues[0],
            'residuals': res.resid
        }
    
    def select_factors(self, max_factors: int = 30) -> pd.DataFrame:

        available_factors = list(self.factors_df.columns)
        selected_factors = []
        results = []
        
        for iteration in range(max_factors):
            best_factor = None
            best_t_stat = 0
            factor_stats = {}
            
            # Préparer les facteurs de base
            if iteration == 0:
                X_base = self.market_return.to_frame()
            else:
                X_base = pd.concat([self.market_return.to_frame()] + 
                                 [self.factors_df[f] for f in selected_factors], axis=1)
            
            # Tester chaque facteur disponible
            for factor in available_factors:
                y = self.factors_df[factor]
                reg_results = self.run_regression(y, X_base)
                factor_stats[factor] = reg_results
                
                if abs(reg_results['t_stat']) > abs(best_t_stat):
                    best_factor = factor
                    best_t_stat = reg_results['t_stat']
            
            if best_factor is None:
                break
                
            # Ajouter le meilleur facteur
            selected_factors.append(best_factor)
            available_factors.remove(best_factor)
            
            # Calculer les statistiques pour les facteurs restants
            if available_factors:
                X_current = pd.concat([self.market_return.to_frame()] + 
                                    [self.factors_df[f] for f in selected_factors], axis=1)
                
                alphas = []
                residuals_list = []
                t_stats = []
                
                for factor in available_factors:
                    y = self.factors_df[factor]
                    reg_results = self.run_regression(y, X_current)
                    alphas.append(reg_results['alpha'])
                    residuals_list.append(reg_results['residuals'])
                    t_stats.append(reg_results['t_stat'])
                
                # Compter les facteurs significatifs
                n_significant_t2 = sum(1 for t in t_stats if abs(t) > 1.96)
                n_significant_t3 = sum(1 for t in t_stats if abs(t) > 3.0)
                
                # Calculer GRS si possible
                if len(alphas) > 0:
                    alphas_array = np.array(alphas)
                    residuals_array = np.column_stack(residuals_list)
                    factors_array = X_current.values[:, 1:]  # Exclure la constante
                    
                    grs_stat, grs_pval = GRSTest.calculate_grs(
                        alphas_array, residuals_array, factors_array
                    )
                    
                    # Calculer avg|alpha|
                    avg_abs_alpha = np.mean(np.abs(alphas))
                    
                    # Calculer Sharpe² ajusté
                    f_bar = np.mean(factors_array, axis=0)
                    Omega = np.cov(factors_array.T)
                    sh2_f = float(f_bar.T @ inv(Omega) @ f_bar)
                else:
                    grs_stat, grs_pval = np.nan, np.nan
                    avg_abs_alpha = 0
                    sh2_f = np.nan
            else:
                n_significant_t2 = 0
                n_significant_t3 = 0
                grs_stat, grs_pval = np.nan, np.nan
                avg_abs_alpha = 0
                sh2_f = np.nan
            
            # Stocker les résultats
            results.append({
                'iteration': iteration + 1,
                'factor': best_factor,
                't_stat': best_t_stat,
                'n_significant_t2': n_significant_t2,
                'n_significant_t3': n_significant_t3,
                'grs_statistic': grs_stat,
                'grs_pvalue': grs_pval,
                'avg_abs_alpha': avg_abs_alpha,
                'sh2_f': sh2_f
            })
            
            # Arrêter si plus de facteurs significatifs
            if n_significant_t3 == 0:
                break
        
        self.results = pd.DataFrame(results)
        return self.results


# 4. FONCTIONS DE TRACÉ
class FactorZooPlotter:
    
    @staticmethod
    def plot_results(results: pd.DataFrame, weighting_scheme: str = 'CW'):

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Résultats de la Compression du Zoo des Facteurs ({weighting_scheme})', 
                     fontsize=16)
        
        # 1. GRS statistic vs k
        ax1 = axes[0, 0]
        ax1.plot(results['iteration'], results['grs_statistic'], 
                'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Nombre de Facteurs (k)')
        ax1.set_ylabel('Statistique GRS')
        ax1.set_title('Statistique GRS vs k')
        ax1.grid(True, alpha=0.3)
        
        # 2. p-value GRS vs k
        ax2 = axes[0, 1]
        ax2.plot(results['iteration'], results['grs_pvalue'], 
                'o-', linewidth=2, markersize=6, color='orange')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% niveau')
        ax2.set_xlabel('Nombre de Facteurs (k)')
        ax2.set_ylabel('p-value GRS')
        ax2.set_title('p-value GRS vs k')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. n(α) vs k pour les deux seuils
        ax3 = axes[1, 0]
        ax3.plot(results['iteration'], results['n_significant_t2'], 
                'o-', linewidth=2, markersize=6, label='t > 1.96', color='blue')
        ax3.plot(results['iteration'], results['n_significant_t3'], 
                's-', linewidth=2, markersize=6, label='t > 3.00', color='red')
        ax3.set_xlabel('Nombre de Facteurs (k)')
        ax3.set_ylabel('Nombre de Facteurs Significatifs')
        ax3.set_title('n(α) vs k')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Sharpe² vs k
        ax4 = axes[1, 1]
        ax4.plot(results['iteration'], results['sh2_f'], 
                'o-', linewidth=2, markersize=6, color='green')
        ax4.set_xlabel('Nombre de Facteurs (k)')
        ax4.set_ylabel('Sharpe² des Facteurs')
        ax4.set_title('Sharpe² vs k')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparison(results_dict: Dict[str, pd.DataFrame]):

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparaison des Schémas de Pondération', fontsize=16)
        
        colors = {'CW': 'blue', 'VW': 'red', 'EW': 'green'}
        
        for scheme, results in results_dict.items():
            color = colors[scheme]
            
            # GRS statistic
            axes[0, 0].plot(results['iteration'], results['grs_statistic'], 
                          'o-', linewidth=2, label=scheme, color=color)
            
            # p-value
            axes[0, 1].plot(results['iteration'], results['grs_pvalue'], 
                          'o-', linewidth=2, label=scheme, color=color)
            
            # n(α) t > 3
            axes[1, 0].plot(results['iteration'], results['n_significant_t3'], 
                          'o-', linewidth=2, label=scheme, color=color)
            
            # Sharpe²
            axes[1, 1].plot(results['iteration'], results['sh2_f'], 
                          'o-', linewidth=2, label=scheme, color=color)
        
        # Configurer les axes
        axes[0, 0].set_title('Statistique GRS')
        axes[0, 0].set_xlabel('Nombre de Facteurs')
        axes[0, 0].set_ylabel('GRS')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('p-value GRS')
        axes[0, 1].set_xlabel('Nombre de Facteurs')
        axes[0, 1].set_ylabel('p-value')
        axes[0, 1].axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Facteurs Significatifs (t > 3)')
        axes[1, 0].set_xlabel('Nombre de Facteurs')
        axes[1, 0].set_ylabel('n(α)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Sharpe² Ajusté')
        axes[1, 1].set_xlabel('Nombre de Facteurs')
        axes[1, 1].set_ylabel('Sharpe²')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


# 5. SCRIPT PRINCIPAL POUR TRAITER LES TROIS SCHÉMAS DE PONDÉRATION
def main():
    
    # Initialiser le chargeur de données
    data_loader = DataLoader()
    
    # Dictionnaire pour stocker les résultats
    results_dict = {}
    
    # Traiter chaque schéma de pondération
    for weighting_scheme in ['CW', 'VW', 'EW']:
        print(f"\nTraitement du schéma de pondération: {weighting_scheme}")
        
        # 1. Charger les données
        factors_df, market_return = data_loader.load_factor_data(weighting_scheme)
        print(f"Données chargées: {factors_df.shape[0]} périodes, {factors_df.shape[1]} facteurs")
        
        # 2. Sélection itérative des facteurs
        selector = IterativeFactorSelection(factors_df, market_return)
        results = selector.select_factors(max_factors=30)
        results_dict[weighting_scheme] = results
        
        # 3. Afficher les résultats
        print(f"\nRésultats pour {weighting_scheme}:")
        print(f"Facteurs nécessaires (t > 3): {results[results['n_significant_t3'] == 0].iloc[0]['iteration']}")
        print(f"p-value GRS > 0.05 à l'itération: {results[results['grs_pvalue'] > 0.05].iloc[0]['iteration'] if any(results['grs_pvalue'] > 0.05) else 'Jamais'}")
        
        # 4. Tracer les résultats individuels
        FactorZooPlotter.plot_results(results, weighting_scheme)
    
    # 5. Comparaison entre les schémas de pondération
    FactorZooPlotter.plot_comparison(results_dict)
    
    # 6. Tableau récapitulatif
    summary_data = []
    for scheme, results in results_dict.items():
        summary_data.append({
            'Scheme': scheme,
            'Factors_t3': results[results['n_significant_t3'] == 0].iloc[0]['iteration'] if any(results['n_significant_t3'] == 0) else '>30',
            'Factors_t2': results[results['n_significant_t2'] == 0].iloc[0]['iteration'] if any(results['n_significant_t2'] == 0) else '>30',
            'GRS_5pct': results[results['grs_pvalue'] > 0.05].iloc[0]['iteration'] if any(results['grs_pvalue'] > 0.05) else '>30'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nTableau récapitulatif:")
    print(summary_df)
    
    return results_dict, summary_df


if __name__ == "__main__":
    results_dict, summary_df = main()
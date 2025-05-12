import numpy as np
import pandas as pd
from scipy.stats import f
from numpy.linalg import inv
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader

class IterativeFactorSelection:

    def __init__(self, factors_df, market_return):
        self.factors_df = factors_df
        self.market_return = market_return
        self.results = []
        
    def run_regression(self, y, X):
        """Effectue une régression OLS et retourne les résultats."""
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const, missing='drop')
        res = model.fit()
        
        return {
            'alpha': res.params[0],
            't_stat': res.tvalues[0],
            'p_value': res.pvalues[0],
            'residuals': res.resid
        }
    
    def calculate_grs(self, alphas, residuals, factors):
        """Calcule la statistique GRS."""
        try:
            T, N = residuals.shape
            K = factors.shape[1] if factors.ndim > 1 else 1
            
            # S'assurer que alphas est un vecteur colonne
            alphas = np.array(alphas).reshape(-1, 1)
            
            # Matrices de covariance
            Sigma = np.cov(residuals.T, bias=False)
            
            # Gérer le cas où factors est 1D
            if factors.ndim == 1:
                factors = factors.reshape(-1, 1)
            
            Omega = np.cov(factors.T, bias=False)
            if Omega.ndim == 0:
                Omega = np.array([[Omega]])
            elif Omega.ndim == 1:
                Omega = Omega.reshape(1, 1)
            
            # Moyenne des facteurs
            f_bar = np.mean(factors, axis=0).reshape(-1, 1)
            
            # Ratios de Sharpe au carré
            Sh2_alpha = float(alphas.T @ inv(Sigma) @ alphas)
            Sh2_f = float(f_bar.T @ inv(Omega) @ f_bar)
            
            # Statistique GRS
            grs_stat = ((T - N - K) / N) * (Sh2_alpha / (1 + Sh2_f))
            
            # p-value
            p_value = 1 - f.cdf(grs_stat, N, T - N - K)
            
            return grs_stat, p_value, Sh2_f
            
        except:
            return np.nan, np.nan, np.nan

    def select_factors_t_std(self, max_factors=30):
        """Sélection itérative des facteurs avec loi de student."""

        available_factors = list(self.factors_df.columns)
        selected_factors = []
        results = []

        # Créer une copie pour la normalisation si nécessaire
        factors_normalized = self.factors_df.copy()
        market_normalized = self.market_return.copy()

        for iteration in range(max_factors):
            print(f"\n--- Itération {iteration + 1} ---")

            best_factor = None
            best_t_stat = 0
            best_alpha = 0

            # Construire le modèle de base
            if iteration == 0:
                # Premier passage : juste le marché
                X_base = market_normalized.to_frame('market')
            else:
                # Passes suivants : marché + facteurs sélectionnés
                X_base = pd.concat([market_normalized.to_frame('market')] +
                                   [factors_normalized[f] for f in selected_factors], axis=1)

            # Tester chaque facteur disponible
            factor_results = {}

            print(f"Test de {len(available_factors)} facteurs disponibles...")

            for factor in available_factors:
                y = factors_normalized[factor]

                # Aligner les données
                valid_idx = ~(y.isna() | X_base.isna().any(axis=1))

                if valid_idx.sum() < 50:  # Au moins 50 observations
                    continue

                reg_results = self.run_regression(y[valid_idx], X_base[valid_idx])

                factor_results[factor] = {
                    'alpha': reg_results['alpha'],
                    't_stat': reg_results['t_stat'],
                    'abs_t_stat': abs(reg_results['t_stat'])
                }

                # Sélectionner le facteur avec le plus grand |t-stat|
                if abs(reg_results['t_stat']) > abs(best_t_stat):
                    best_factor = factor
                    best_t_stat = reg_results['t_stat']
                    best_alpha = reg_results['alpha']

            # Afficher les top 5 facteurs pour cette itération
            sorted_factors = sorted(factor_results.items(),
                                    key=lambda x: x[1]['abs_t_stat'],
                                    reverse=True)

            print(f"\nTop 5 facteurs (itération {iteration + 1}):")
            for i, (fac, res) in enumerate(sorted_factors[:5]):
                print(f"  {i + 1}. {fac:<20} alpha: {res['alpha']:8.4f}, t-stat: {res['t_stat']:8.2f}")

            if best_factor is None:
                print("Aucun facteur sélectionné - arrêt")
                break

            # Ajouter le meilleur facteur
            selected_factors.append(best_factor)
            available_factors.remove(best_factor)

            print(f"\nFacteur sélectionné: {best_factor} (t-stat: {best_t_stat:.3f})")

            # Calculer les statistiques pour ce modèle
            X_current = pd.concat([market_normalized.to_frame('market')] +
                                  [factors_normalized[f] for f in selected_factors], axis=1)

            # Statistiques des facteurs restants
            if available_factors:
                alphas = []
                residuals_list = []
                t_stats = []

                for factor in available_factors:
                    y = factors_normalized[factor]
                    valid_idx = ~(y.isna() | X_current.isna().any(axis=1))

                    if valid_idx.sum() < 50:
                        continue

                    reg_results = self.run_regression(y[valid_idx], X_current[valid_idx])
                    alphas.append(reg_results['alpha'])
                    residuals_list.append(reg_results['residuals'])
                    t_stats.append(reg_results['t_stat'])

                # Compter les facteurs significatifs
                n_significant_t2 = sum(1 for t in t_stats if abs(t) > 1.96)
                n_significant_t3 = sum(1 for t in t_stats if abs(t) > 3.0)

                # Calculer GRS
                if len(alphas) > 0 and len(residuals_list) > 0:
                    # S'assurer que tous les résidus ont la même longueur
                    min_length = min(len(res) for res in residuals_list)
                    residuals_array = np.column_stack([res[:min_length] for res in residuals_list])

                    # Facteurs pour GRS (sans la constante)
                    factors_array = X_current.iloc[:min_length, 1:].values

                    grs_stat, grs_pval, sh2_f = self.calculate_grs(
                        np.array(alphas), residuals_array, factors_array
                    )

                    avg_abs_alpha = np.mean(np.abs(alphas))
                else:
                    grs_stat, grs_pval, sh2_f = np.nan, np.nan, np.nan
                    avg_abs_alpha = 0
            else:
                n_significant_t2, n_significant_t3 = 0, 0
                grs_stat, grs_pval, sh2_f = np.nan, np.nan, np.nan
                avg_abs_alpha = 0

            # Calculer le Sharpe ratio
            if X_current.shape[1] > 1:
                # Calculer rendements moyens et Sharpe ratio
                factor_returns = X_current.iloc[:, 1:].mean()
                factor_std = X_current.iloc[:, 1:].std()
                sharpe_ratio = (factor_returns.mean() / factor_std.mean()) * np.sqrt(12)  # Annualisé
            else:
                sharpe_ratio = 0

            # Stocker les résultats
            results.append({
                'iteration': iteration + 1,
                'factor': best_factor,
                'alpha': best_alpha,
                't_stat': best_t_stat,
                'n_significant_t2': n_significant_t2,
                'n_significant_t3': n_significant_t3,
                'grs_statistic': grs_stat,
                'grs_pvalue': grs_pval,
                'avg_abs_alpha': avg_abs_alpha * 12 * 100,  # Annualisé en %
                'sh2_f': sh2_f,
                'sr': sharpe_ratio
            })

            print(f"Facteurs significatifs restants (t > 3): {n_significant_t3}")
            print(f"GRS statistic: {grs_stat:.3f}, p-value: {grs_pval:.3f}")

            '''# Arrêter si plus de facteurs significatifs
            if n_significant_t3 == 0:
                print(f"\nArrêt à l'itération {iteration + 1}: Plus de facteurs significatifs avec t > 3.0")
                break'''

        self.results = pd.DataFrame(results)

        # Formater les colonnes pour correspondre à l'article
        self.results['GRS'] = self.results['grs_statistic'].round(2)
        self.results['p(GRS)'] = self.results['grs_pvalue'].round(2)
        self.results['Avg|α|'] = self.results['avg_abs_alpha'].round(2)
        self.results['Sh²(f)'] = self.results['sh2_f'].round(2)
        self.results['SR'] = self.results['sr'].round(2)

        return self.results

    def select_factors_GRS(self, max_factors=30):
        """Implémentation de la sélection itérative des facteurs selon la méthodologie de l'article (GRS)."""
        
        available_factors = list(self.factors_df.columns)
        selected_factors = []
        results = []
        
        factors_normalized = self.factors_df.copy()
        market_normalized = self.market_return.copy()
        
        for iteration in range(max_factors):
            print(f"\n--- Itération {iteration + 1} ---")
            
            # Construire le modèle de base pour cette itération
            if iteration == 0:
                # Premier passage : juste le marché (CAPM)
                X_base = market_normalized.to_frame('market')
            else:
                # Passes suivantes : marché + facteurs déjà sélectionnés
                X_base = pd.concat([market_normalized.to_frame('market')] + 
                                 [factors_normalized[f].to_frame(f) for f in selected_factors], axis=1)
            
            # Tester chaque facteur candidat disponible
            best_factor = None
            best_grs = float('inf')  # On cherche à minimiser la statistique GRS
            best_factor_results = {}
            
            print(f"Test de {len(available_factors)} facteurs candidats...")
            
            for factor in available_factors:
                # Créer le modèle augmenté avec ce facteur candidat
                X_augmented = pd.concat([X_base, factors_normalized[factor].to_frame(factor)], axis=1)
                
                # Calculer les alphas et résidus pour tous les autres facteurs par rapport à ce modèle
                alphas = []
                residuals_list = []
                t_stats = []
                
                for test_factor in [f for f in available_factors if f != factor]:
                    y = factors_normalized[test_factor]
                    
                    # Aligner les données
                    valid_idx = ~(y.isna() | X_augmented.isna().any(axis=1))
                    
                    if valid_idx.sum() < 60:  # Minimum d'observations
                        continue
                    
                    reg_results = self.run_regression(y[valid_idx], X_augmented[valid_idx])
                    alphas.append(reg_results['alpha'])
                    residuals_list.append(reg_results['residuals'])
                    t_stats.append(reg_results['t_stat'])
                
                # S'assurer qu'il y a suffisamment de données pour le calcul GRS
                if len(alphas) == 0 or len(residuals_list) == 0:
                    continue
                    
                # Préparer les données pour le calcul GRS
                min_length = min(len(res) for res in residuals_list)
                residuals_array = np.column_stack([res[:min_length] for res in residuals_list])
                factors_array = X_augmented.iloc[:min_length, :].values
                
                # Calculer la statistique GRS pour ce modèle augmenté
                grs_stat, grs_pval, sh2_f = self.calculate_grs(
                    np.array(alphas), residuals_array, factors_array
                )
                
                # Compter les facteurs significatifs restants
                n_significant = sum(1 for t in t_stats if abs(t) > self.significance_threshold)
                
                # Stocker les résultats pour ce facteur
                factor_results = {
                    'grs': grs_stat,
                    'p_value': grs_pval,
                    'sh2_f': sh2_f,
                    'n_significant': n_significant,
                    'avg_abs_alpha': np.mean(np.abs(alphas)) * 12 * 100,  # Annualisé en %
                    't_stats': t_stats
                }
                
                print(f"  {factor:<20} GRS: {grs_stat:8.3f}, p-value: {grs_pval:8.3f}, Facteurs sig.: {n_significant}")
                
                # Sélectionner le facteur avec la statistique GRS la plus faible
                if np.isfinite(grs_stat) and grs_stat < best_grs:
                    best_grs = grs_stat
                    best_factor = factor
                    best_factor_results = factor_results
            
            # Vérifier si un facteur a été trouvé
            if best_factor is None:
                print("Aucun facteur valide trouvé - arrêt")
                break
            
            # Ajouter le meilleur facteur au modèle
            selected_factors.append(best_factor)
            available_factors.remove(best_factor)
            
            print(f"\nFacteur sélectionné: {best_factor} (GRS: {best_grs:.3f})")
            
            # Stocker les résultats de cette itération
            results.append({
                'iteration': iteration + 1,
                'factor': best_factor,
                'grs_statistic': best_grs,
                'grs_pvalue': best_factor_results['p_value'],
                'n_significant_t3': best_factor_results['n_significant'],
                'avg_abs_alpha': best_factor_results['avg_abs_alpha'],
                'sh2_f': best_factor_results['sh2_f']
            })
            
            # Critère d'arrêt: plus de facteurs significatifs
            if best_factor_results['n_significant'] == 0:
                print(f"\nArrêt à l'itération {iteration+1}: Plus de facteurs significatifs avec t > {self.significance_threshold}")
                break
        
        # Préparer le DataFrame des résultats
        self.results = pd.DataFrame(results)
        
        # Formater les colonnes pour correspondre à l'article
        if not self.results.empty:
            self.results['GRS'] = self.results['grs_statistic'].round(2)
            self.results['p(GRS)'] = self.results['grs_pvalue'].round(2)
            self.results['Avg|α|'] = self.results['avg_abs_alpha'].round(2)
            self.results['Sh²(f)'] = self.results['sh2_f'].round(2)
            
            # Calculer Sharpe ratio pour chaque modèle
            for i in range(len(self.results)):
                factors_in_model = selected_factors[:i+1]
                X_model = pd.concat([market_normalized.to_frame('market')] + 
                                  [factors_normalized[f].to_frame(f) for f in factors_in_model], axis=1)
                
                # Calculer Sharpe ratio annualisé
                if X_model.shape[1] > 1:
                    factor_returns = X_model.iloc[:, 1:].mean()
                    factor_std = X_model.iloc[:, 1:].std()
                    sr = (factor_returns.mean() / factor_std.mean()) * np.sqrt(12)
                    self.results.loc[i, 'SR'] = round(sr, 2)
                else:
                    self.results.loc[i, 'SR'] = 0.0
        
        return self.results
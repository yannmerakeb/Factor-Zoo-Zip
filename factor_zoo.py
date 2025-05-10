import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
from numpy.linalg import inv
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Chargement et nettoyage des données de facteurs."""
    def __init__(self, weighting, start_date="1971-11-30", end_date="2021-12-31"):
        self.data_path = f'data/{weighting}.csv'
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def load_factor_data(self, region = 'world'):
        """Charge les données des facteurs et extrait le rendement du marché."""
        print(f"Chargement des données depuis {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        if region == 'US':
            df = df[df['location'] == 'US']
        elif region == 'ex US':
            df = df[df['location'] != 'US']

        # Pivot et extraction du marché
        # à modifier pour plusieurs régions
        pivot_df = df.pivot(index='date', columns='name', values='ret')
        market_return = pivot_df['market_equity']
        factors_df = pivot_df.drop(columns=['market_equity'])
        
        print(f"Données chargées: {len(factors_df)} périodes, {factors_df.shape[1]} facteurs")
        return factors_df, market_return


class GRSTest:
    """Calcul de la statistique GRS et de sa p-value."""
    @staticmethod
    def calculate_grs(alphas, residuals, factors):
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
        grs_stat = ((T - N - K) / N) * (Sh2_alpha / (1 + Sh2_f))
        
        # p-value
        p_value = 1 - f.cdf(grs_stat, N, T - N - K)
        
        return grs_stat, p_value, Sh2_f


class IterativeFactorSelection:
    
    def __init__(self, factors_df, market_return, significance_threshold=3.0):
        self.factors_df = factors_df
        self.market_return = market_return
        self.significance_threshold = significance_threshold
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
    
    def select_factors(self, max_factors=30):
        """Sélection itérative des facteurs."""
        
        available_factors = list(self.factors_df.columns)
        selected_factors = []
        results = []
        
        # Créer une copie pour la normalisation si nécessaire
        factors_normalized = self.factors_df.copy()
        market_normalized = self.market_return.copy()
        
        # Vérifier et normaliser les données si nécessaire
        if abs(market_normalized.mean()) > 0.05:
            print("Détection de données en pourcentage - conversion en décimal")
            factors_normalized = factors_normalized / 100
            market_normalized = market_normalized / 100
        
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
                print(f"  {i+1}. {fac:<20} alpha: {res['alpha']:8.4f}, t-stat: {res['t_stat']:8.2f}")
            
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
            
            # Arrêter si plus de facteurs significatifs
            if n_significant_t3 == 0:
                print(f"\nArrêt à l'itération {iteration+1}: Plus de facteurs significatifs avec t > 3.0")
                break
        
        self.results = pd.DataFrame(results)
        
        # Formater les colonnes pour correspondre à l'article
        self.results['GRS'] = self.results['grs_statistic'].round(2)
        self.results['p(GRS)'] = self.results['grs_pvalue'].round(2)
        self.results['Avg|α|'] = self.results['avg_abs_alpha'].round(2)
        self.results['Sh²(f)'] = self.results['sh2_f'].round(2)
        self.results['SR'] = self.results['sr'].round(2)
        
        return self.results

# Test de diagnostic pour vérifier les données
def diagnostic_check(factors_df, market_return):
    """Vérifie les données et suggère des corrections."""
    
    print("DIAGNOSTIC DES DONNÉES")
    print("=" * 50)
    
    # Vérifier le marché
    print(f"Marché - Moyenne: {market_return.mean():.6f}")
    print(f"Marché - Écart-type: {market_return.std():.6f}")
    
    # Vérifier quelques facteurs clés
    key_factors = ['cop_at', 'noa_gr1a', 'saleq_gr1', 'ival_me', 'resff3_12_1']
    
    for factor in key_factors:
        if factor in factors_df.columns:
            print(f"\n{factor}:")
            print(f"  Moyenne: {factors_df[factor].mean():.6f}")
            print(f"  Écart-type: {factors_df[factor].std():.6f}")
            
            # Test de régression simple
            y = factors_df[factor]
            X = sm.add_constant(market_return)
            valid_idx = ~(y.isna() | market_return.isna())
            
            model = sm.OLS(y[valid_idx], X[valid_idx])
            results = model.fit()
            
            print(f"  Alpha CAPM: {results.params[0]:.6f}")
            print(f"  t-stat: {results.tvalues[0]:.3f}")
            print(f"  Alpha annualisé: {results.params[0] * 12 * 100:.2f}%")
    
    # Recommandation
    if abs(market_return.mean()) > 0.05:
        print("\n⚠️ Les données semblent être en pourcentage!")
        print("Recommandation: Diviser par 100 avant analyse")
    else:
        print("\n✓ Les données semblent être en décimal")
    
    return {
        'market_mean': market_return.mean(),
        'market_std': market_return.std(),
        'likely_percentage': abs(market_return.mean()) > 0.05
    }

# Fonction pour créer une comparaison directe avec l'article
def create_exhibit_comparison(your_results, formatted=True):
    """Crée un tableau comparable à l'Exhibit 2 de l'article."""
    
    # Structure similaire à l'article
    exhibit = pd.DataFrame()
    
    exhibit['No.'] = your_results['iteration']
    exhibit['Factor'] = your_results['factor']
    
    # Clusters (vous devez les mapper)
    from clusters import create_factor_clusters
    get_cluster = create_factor_clusters()
    exhibit['Cluster'] = your_results['factor'].apply(get_cluster)
    
    # Statistiques
    exhibit['GRS'] = your_results['grs_statistic'].round(2)
    exhibit['p(GRS)'] = your_results['grs_pvalue'].round(3)
    exhibit['Avg|α|'] = your_results['avg_abs_alpha'].round(2)
    exhibit['Sh²(f)'] = your_results['sh2_f'].round(2)
    exhibit['SR'] = your_results['sr'].round(2)
    exhibit['t > 2'] = your_results['n_significant_t2']
    exhibit['t > 3'] = your_results['n_significant_t3']
    
    if formatted:
        # Formater comme dans l'article
        exhibit['GRS'] = exhibit['GRS'].map('{:.2f}'.format)
        exhibit['p(GRS)'] = exhibit['p(GRS)'].map('{:.2f}'.format)
        exhibit['Avg|α|'] = exhibit['Avg|α|'].map('{:.2f}'.format)
        exhibit['Sh²(f)'] = exhibit['Sh²(f)'].map('{:.2f}'.format)
        exhibit['SR'] = exhibit['SR'].map('{:.2f}'.format)
    
    return exhibit
class FactorZooPlotter:
    """Visualisation des résultats de la sélection des facteurs."""
    
    @staticmethod
    def create_exhibit_table(results_dict):
        """Crée un tableau similaire à l'Exhibit 3 de l'article."""
        
        all_factors = []
        for scheme, results in results_dict.items():
            top_factors = []
            # Extraire les 15 premiers facteurs sélectionnés
            for i in range(min(15, len(results))):
                factor_info = {
                    'No.': i+1,
                    'Factor': results.iloc[i]['factor'],
                    'Scheme': scheme,
                    't_stat': results.iloc[i]['t_stat'],
                    'GRS': results.iloc[i]['grs_statistic'],
                    'p(GRS)': results.iloc[i]['grs_pvalue'],
                    'n_t>2': results.iloc[i]['n_significant_t2'],
                    'n_t>3': results.iloc[i]['n_significant_t3']
                }
                top_factors.append(factor_info)
            all_factors.extend(top_factors)
        
        # Créer le DataFrame
        factor_table = pd.DataFrame(all_factors)
        
        # Formater pour l'affichage
        factor_table['GRS'] = factor_table['GRS'].map('{:.2f}'.format)
        factor_table['p(GRS)'] = factor_table['p(GRS)'].map('{:.2f}'.format)
        factor_table['t_stat'] = factor_table['t_stat'].map('{:.2f}'.format)
        
        return factor_table
    
    @staticmethod
    def plot_comparison(results_dict):
        """Graphique de comparaison des schémas de pondération, similaire à l'exhibit n°7."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparaison des Schémas de Pondération', fontsize=16)
        
        colors = {'CW': 'blue', 'VW': 'red', 'EW': 'green'}
        
        for scheme, results in results_dict.items():
            color = colors.get(scheme, 'black')
            
            # Tracer les quatre graphiques
            axes[0, 0].plot(results['iteration'], results['grs_statistic'], 'o-', linewidth=2, label=scheme, color=color)
            axes[0, 1].plot(results['iteration'], results['grs_pvalue'], 'o-', linewidth=2, label=scheme, color=color)
            axes[1, 0].plot(results['iteration'], results['n_significant_t3'], 'o-', linewidth=2, label=scheme, color=color)
            axes[1, 1].plot(results['iteration'], results['sh2_f'], 'o-', linewidth=2, label=scheme, color=color)
        
        # Configuration des axes
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
    
    def create_summary_table(results_dict):
        """Crée un tableau récapitulatif similaire à celui mentionné dans l'article."""
        import pandas as pd
        
        summary_data = []
        for scheme, results in results_dict.items():
            # Premier facteur où il n'y a plus de facteurs significatifs avec t > 3
            t3_iterations = results[results['n_significant_t3'] == 0]
            t3_iter = t3_iterations.iloc[0]['iteration'] if not t3_iterations.empty else '>30'
            
            # Premier facteur où il n'y a plus de facteurs significatifs avec t > 2
            t2_iterations = results[results['n_significant_t2'] == 0]
            t2_iter = t2_iterations.iloc[0]['iteration'] if not t2_iterations.empty else '>30'
            
            # Premier facteur où GRS n'est plus significatif (p > 0.05)
            grs_iterations = results[results['grs_pvalue'] > 0.05]
            grs_iter = grs_iterations.iloc[0]['iteration'] if not grs_iterations.empty else '>30'
            
            # Nombre total de facteurs testés
            total_factors = len(results) if not results.empty else 0
            
            summary_data.append({
                'Weighting Scheme': scheme,
                'Total Factors': total_factors,
                'No. Factors (t > 3)': t3_iter,
                'No. Factors (t > 2)': t2_iter,
                'No. Factors (GRS 5%)': grs_iter
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Formater pour l'affichage
        return summary_df.set_index('Weighting Scheme')
    
    @staticmethod
    def plot_alphas_chart(factors_df, selected_factors, clusters, market_return=None):
        """Crée un graphique similaire à l'Exhibit 4 de l'article montrant les alphas des facteurs."""
        
        # Calculer les alphas CAPM pour tous les facteurs
        alphas = {}
        
        # Si market_return n'est pas fourni, utiliser la moyenne des facteurs comme proxy
        if market_return is None:
            market_return = factors_df.mean(axis=1)
        
        # Calculer l'alpha pour chaque facteur
        for factor in factors_df.columns:
            y = factors_df[factor]
            X = sm.add_constant(market_return)
            model = sm.OLS(y, X, missing='drop').fit()
            alphas[factor] = model.params[0] * 100  # Convertir en pourcentage
        
        alpha_df = pd.DataFrame.from_dict(alphas, orient='index', columns=['alpha'])
        alpha_df['cluster'] = pd.Series(clusters)
        
        # Déterminer si un facteur est sélectionné
        alpha_df['selected'] = alpha_df.index.isin(selected_factors)
        
        # Trier par cluster puis par alpha
        alpha_df = alpha_df.sort_values(['cluster', 'alpha'])
        
        # Configuration du graphique
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Définir les couleurs pour chaque cluster
        cluster_colors = {
            'Quality': '#8B0000',        # Dark red
            'Investment': '#B22222',     # Firebrick
            'Low Risk': '#90EE90',       # Light green
            'Value': '#1E90FF',          # Dodger blue
            'Momentum': '#00BFFF',       # Deep sky blue
            'Seasonality': '#000080',    # Navy
            'Profitability': '#8B4513',  # Saddle brown
            'Debt Issuance': '#FFA500',  # Orange
            'Low Leverage': '#483D8B',   # Dark slate blue
            'Profit Growth': '#006400',  # Dark green
            'Short-Term Reversal': '#556B2F', # Dark olive green
            'Accruals': '#20B2AA',       # Light sea green
            'Size': '#00CED1',           # Dark turquoise
            'Market': '#808080'          # Gray
        }
        
        # Tracer les barres par cluster
        bars = []
        x_positions = []
        x_pos = 0
        
        for cluster in alpha_df['cluster'].unique():
            cluster_df = alpha_df[alpha_df['cluster'] == cluster]
            
            # Tracer les facteurs non sélectionnés en blanc transparent
            non_selected = cluster_df[~cluster_df['selected']]
            selected = cluster_df[cluster_df['selected']]
            
            # Barres non sélectionnées
            for i, (idx, row) in enumerate(non_selected.iterrows()):
                bar = ax.bar(x_pos, row['alpha'], width=0.7, 
                       color='white', edgecolor='lightgrey', alpha=0.5)
                x_pos += 1
            
            # Barres sélectionnées
            for i, (idx, row) in enumerate(selected.iterrows()):
                bar = ax.bar(x_pos, row['alpha'], width=0.7, 
                       color=cluster_colors.get(cluster, '#333333'), alpha=0.8)
                bars.append(bar[0])
                x_positions.append(x_pos)
                x_pos += 1
            
            # Ajouter un espace entre les clusters
            x_pos += 1
        
        # Ajouter une légende pour les clusters
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8) 
                           for color in cluster_colors.values()]
        ax.legend(legend_elements, cluster_colors.keys(), 
                  loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=7, frameon=False)
        
        # Labels et titre
        ax.set_title('Selected Alpha Factors', fontsize=14)
        ax.set_ylabel('Alpha [%]', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Supprimer les étiquettes de l'axe x car il y a trop de facteurs
        ax.set_xticks([])
        
        plt.tight_layout()
        return fig
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f
from numpy.linalg import inv
import statsmodels.api as sm
import warnings
from data_loader import DataLoader

warnings.filterwarnings('ignore')


class RPPCA:
    """
    Classe pour l'analyse en composantes principales avec prime de risque (RP-PCA).
    Inclut une sélection itérative des composantes principales basée sur la t-statistique de l'alpha.
    Adaptée pour des données sous forme de DataFrame (facteurs) et Series (marché).
    """

    def __init__(self, factors_df, market_return, lambda_reg=0.5, n_components=None, standardize=True):
        """
        Initialisation de la classe RP-PCA.

        Paramètres :
        - factors_df (pd.DataFrame) : DataFrame des rendements des facteurs (T x N).
        - market_return (pd.Series) : Series des rendements excédentaires du marché (T x 1).
        - lambda_reg (float) : Paramètre de régularisation pour équilibrer variance et rendements moyens.
        - n_components (int) : Nombre de composantes principales à extraire (si None, toutes).
        - standardize (bool) : Standardiser les données avant l'analyse (recommandé).
        """
        self.factors_df = factors_df
        self.market_return = market_return
        self.lambda_reg = lambda_reg
        self.n_components = n_components
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.mean_returns = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.risk_premium_weights_ = None
        self.component_scores_ = None
        self.selection_results_ = None

    def fit(self):
        """
        Ajuste le modèle RP-PCA sur les données des facteurs.

        Retourne :
        - self : Instance ajustée de la classe.
        """
        # Convertir le DataFrame en numpy array
        X = self.factors_df.values

        # Vérifier l'échelle des données (rendements en décimal, pas en pourcentage)
        mean_returns_raw = np.mean(X, axis=0)
        max_mean_abs = np.max(np.abs(mean_returns_raw))
        if max_mean_abs > 0.1:  # Si les rendements moyens sont trop grands (probablement en pourcentage)
            print(
                f"ATTENTION : Rendements moyens trop élevés (max |mean| = {max_mean_abs:.4f}). Conversion en décimal...")
            X = X / 100
            self.factors_df = self.factors_df / 100  # Mettre à jour le DataFrame

        # Standardisation des données si nécessaire (désactivée par défaut)
        if self.standardize:
            print("Standardisation activée : les rendements moyens seront centrés à zéro.")
            X = self.scaler.fit_transform(X)

        # Calculer les rendements moyens (premier moment)
        self.mean_returns = np.mean(X, axis=0)
        print("Rendements moyens des facteurs :", self.mean_returns)

        # Vérifier si les rendements moyens sont proches de zéro
        if np.all(np.abs(self.mean_returns) < 1e-10):
            print("AVERTISSEMENT : Rendements moyens proches de zéro. Vérifiez la standardisation ou les données.")

        # Calculer la matrice de covariance (second moment)
        cov_matrix = np.cov(X.T)
        print("Matrice de covariance (diagonale) :", np.diag(cov_matrix))

        # Ajuster la matrice pour inclure les rendements moyens (RP-PCA)
        mu_outer = np.outer(self.mean_returns, self.mean_returns)
        print("Matrice mu mu^T (max) :", np.max(mu_outer))
        adjusted_matrix = cov_matrix + self.lambda_reg * mu_outer

        # Décomposition en valeurs propres de la matrice ajustée
        eigenvalues, eigenvectors = np.linalg.eigh(adjusted_matrix)

        # Trier les valeurs propres et vecteurs propres par ordre décroissant
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Sélectionner le nombre de composantes
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components_ = eigenvectors[:, :self.n_components]

        # Calculer la variance expliquée
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        print("Proportion de variance expliquée :", self.explained_variance_ratio_)

        # Poids des primes de risque
        self.risk_premium_weights_ = np.dot(self.mean_returns, self.components_)
        print("Poids des primes de risque :", self.risk_premium_weights_)

        # Calculer les scores des composantes (facteurs latents)
        self.component_scores_ = np.dot(X, self.components_)
        self.component_scores_ = pd.DataFrame(
            self.component_scores_,
            index=self.factors_df.index,
            columns=[f"PC{i + 1}" for i in range(self.n_components)]
        )

        return self

    def transform(self, X):
        """
        Transforme les données en utilisant les composantes RP-PCA.

        Paramètres :
        - X (pd.DataFrame ou np.ndarray) : Données à transformer.

        Retourne :
        - X_transformed : Données projetées sur les composantes principales.
        """
        if self.components_ is None:
            raise ValueError("Le modèle doit être ajusté avant la transformation.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.standardize:
            X = self.scaler.transform(X)

        X_transformed = np.dot(X, self.components_)
        return X_transformed

    def fit_transform(self):
        """
        Ajuste le modèle et transforme les données en une seule étape.

        Retourne :
        - X_transformed : Données projetées.
        """
        return self.fit().transform(self.factors_df)

    def get_components(self):
        """
        Retourne les composantes principales (facteurs latents).

        Retourne :
        - components : Matrice des vecteurs propres (N x n_components).
        """
        return self.components_

    def get_explained_variance_ratio(self):
        """
        Retourne la proportion de variance expliquée par chaque composante.

        Retourne :
        - explained_variance_ratio : Array des ratios de variance expliquée.
        """
        return self.explained_variance_ratio_

    def get_risk_premium_weights(self):
        """
        Retourne les poids des primes de risque pour chaque composante.

        Retourne :
        - risk_premium_weights : Array des projections des rendements moyens.
        """
        return self.risk_premium_weights_

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

            alphas = np.array(alphas).reshape(-1, 1)
            Sigma = np.cov(residuals.T, bias=False)

            if factors.ndim == 1:
                factors = factors.reshape(-1, 1)

            Omega = np.cov(factors.T, bias=False)
            if Omega.ndim == 0:
                Omega = np.array([[Omega]])
            elif Omega.ndim == 1:
                Omega = Omega.reshape(1, 1)

            f_bar = np.mean(factors, axis=0).reshape(-1, 1)
            Sh2_alpha = float(alphas.T @ inv(Sigma) @ alphas)
            Sh2_f = float(f_bar.T @ inv(Omega) @ f_bar)

            grs_stat = ((T - N - K) / N) * (Sh2_alpha / (1 + Sh2_f))
            p_value = 1 - f.cdf(grs_stat, N, T - N - K)

            return grs_stat, p_value, Sh2_f

        except:
            return np.nan, np.nan, np.nan

    def select_components_t_stat(self, max_components=10):
        """
        Sélection itérative des composantes principales basée sur la t-statistique de l'alpha.

        Paramètres :
        - max_components (int) : Nombre maximum de composantes à sélectionner.
        - significance_threshold (float) : Seuil pour les t-statistiques (par défaut 3.0).

        Retourne :
        - results : DataFrame contenant les résultats de la sélection.
        """
        if self.component_scores_ is None:
            raise ValueError("Le modèle RP-PCA doit être ajusté avant la sélection.")

        available_components = list(self.component_scores_.columns)
        selected_components = []
        results = []

        factors_normalized = self.component_scores_.copy()
        market_normalized = self.market_return.copy()

        for iteration in range(max_components):
            print(f"\n--- Itération {iteration + 1} ---")

            best_component = None
            best_t_stat = 0
            best_alpha = 0

            if iteration == 0:
                X_base = market_normalized.to_frame('market')
            else:
                X_base = pd.concat([market_normalized.to_frame('market')] +
                                   [factors_normalized[c].to_frame(c) for c in selected_components], axis=1)

            factor_results = {}
            print(f"Test de {len(available_components)} composantes disponibles...")

            for component in available_components:
                y = factors_normalized[component]
                valid_idx = ~(y.isna() | X_base.isna().any(axis=1))

                if valid_idx.sum() < 50:
                    continue

                reg_results = self.run_regression(y[valid_idx], X_base[valid_idx])

                factor_results[component] = {
                    'alpha': reg_results['alpha'],
                    't_stat': reg_results['t_stat'],
                    'abs_t_stat': abs(reg_results['t_stat'])
                }

                if abs(reg_results['t_stat']) > abs(best_t_stat):
                    best_component = component
                    best_t_stat = reg_results['t_stat']
                    best_alpha = reg_results['alpha']

            sorted_factors = sorted(factor_results.items(),
                                    key=lambda x: x[1]['abs_t_stat'],
                                    reverse=True)

            print(f"\nTop 5 composantes (itération {iteration + 1}):")
            for i, (comp, res) in enumerate(sorted_factors[:5]):
                print(f"  {i + 1}. {comp:<20} alpha: {res['alpha']:8.4f}, t-stat: {res['t_stat']:8.2f}")

            if best_component is None:
                print("Aucune composante sélectionnée - arrêt")
                break

            selected_components.append(best_component)
            available_components.remove(best_component)

            print(f"\nComposante sélectionnée: {best_component} (t-stat: {best_t_stat:.3f})")

            X_current = pd.concat([market_normalized.to_frame('market')] +
                                  [factors_normalized[c].to_frame(c) for c in selected_components], axis=1)

            if available_components:
                alphas = []
                residuals_list = []
                t_stats = []

                for component in available_components:
                    y = factors_normalized[component]
                    valid_idx = ~(y.isna() | X_current.isna().any(axis=1))

                    if valid_idx.sum() < 50:
                        continue

                    reg_results = self.run_regression(y[valid_idx], X_current[valid_idx])
                    alphas.append(reg_results['alpha'])
                    residuals_list.append(reg_results['residuals'])
                    t_stats.append(reg_results['t_stat'])

                n_significant_t2 = sum(1 for t in t_stats if abs(t) > 1.96)
                n_significant_t3 = sum(1 for t in t_stats if abs(t) > 3.0)

                if len(alphas) > 0 and len(residuals_list) > 0:
                    min_length = min(len(res) for res in residuals_list)
                    residuals_array = np.column_stack([res[:min_length] for res in residuals_list])
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

            if X_current.shape[1] > 1:
                factor_returns = X_current.iloc[:, 1:].mean()
                factor_std = X_current.iloc[:, 1:].std()
                sharpe_ratio = (factor_returns.mean() / factor_std.mean()) * np.sqrt(12)
            else:
                sharpe_ratio = 0

            results.append({
                'iteration': iteration + 1,
                'component': best_component,
                'alpha': best_alpha,
                't_stat': best_t_stat,
                'n_significant_t2': n_significant_t2,
                'n_significant_t3': n_significant_t3,
                'grs_statistic': grs_stat,
                'grs_pvalue': grs_pval,
                'avg_abs_alpha': avg_abs_alpha * 12 * 100,
                'sh2_f': sh2_f,
                'sr': sharpe_ratio
            })

            print(f"Composantes significatives restantes (t > 3): {n_significant_t3}")
            print(f"GRS statistic: {grs_stat:.3f}, p-value: {grs_pval:.3f}")

            if n_significant_t3 == 0:
                print(f"\nArrêt à l'itération {iteration + 1}: Plus de composantes significatives avec t > 3.0")
                break

        self.selection_results_ = pd.DataFrame(results)

        if not self.selection_results_.empty:
            self.selection_results_['GRS'] = self.selection_results_['grs_statistic'].round(2)
            self.selection_results_['p(GRS)'] = self.selection_results_['grs_pvalue'].round(2)
            self.selection_results_['Avg|α|'] = self.selection_results_['avg_abs_alpha'].round(2)
            self.selection_results_['Sh²(f)'] = self.selection_results_['sh2_f'].round(2)
            self.selection_results_['SR'] = self.selection_results_['sr'].round(2)

        return self.selection_results_


if __name__ == "__main__":  # MODIFIER : Si vos données sont dans un autre format, ajustez (ex. pd.read_excel)

    # Charger les données
    data_loader = DataLoader('VW_cap', '1993-08-01', '2021-12-31')
    data_loader.data_path = f'../data/{data_loader.weighting}.csv'
    factors_df, market_return = data_loader.load_factor_data(region='US')

    # MODIFICATION : Vérifier l'alignement des données
    # S'assurer que factors_df et market_return ont le même index
    common_index = factors_df.index.intersection(market_return.index)
    factors_df = factors_df.loc[common_index]
    market_return = market_return.loc[common_index]
    print(f"Données alignées : {factors_df.shape[0]} observations")

    # Initialiser et ajuster le modèle RP-PCA
    rp_pca = RPPCA(factors_df, market_return, lambda_reg=10, n_components=10, standardize=True)
    rp_pca.fit()

    # Afficher les résultats de la RP-PCA
    print("\nComposantes principales (facteurs latents) :")
    print(pd.DataFrame(rp_pca.get_components(), index=factors_df.columns))
    print("\nProportion de variance expliquée :")
    print(rp_pca.get_explained_variance_ratio())
    print("\nPoids des primes de risque :")
    print(rp_pca.get_risk_premium_weights())

    # Effectuer la sélection itérative des composantes basée sur la t-statistique
    # MODIFICATION : Ajustez max_components et significance_threshold si nécessaire
    # - max_components : Nombre maximum de composantes à tester
    # - significance_threshold : Seuil pour considérer une composante comme significative (ex. 3.0)
    selection_results = rp_pca.select_components_t_stat(max_components=10)
    print("\nRésultats de la sélection itérative des composantes :")
    print(selection_results)
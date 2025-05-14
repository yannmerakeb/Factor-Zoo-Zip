import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import Rotator


from data_loader import DataLoader

class PCAPurification:
    """
    Extension du Factor Zoo qui utilise l'ACP pour:
    1. Identifier les facteurs principaux sous-jacents
    2. Purifier les facteurs existants de leurs expositions communes
    3. Créer de nouveaux méta-facteurs orthogonaux
    """
    
    def __init__(self, factors_df, market_return=None):
        """
        Initialise la classe avec les données des facteurs.
        
        Args:
            factors_df (pd.DataFrame): DataFrame contenant les rendements des facteurs
            market_return (pd.Series, optional): Rendement du marché, si disponible
        """
        self.factors_df = factors_df.copy()
        self.market_return = market_return
        self.n_factors = factors_df.shape[1]
        self.factor_names = factors_df.columns.tolist()
        self.pca_results = None
        self.explained_variance = None
        self.loadings = None
        self.principal_factors = None
        self.purified_factors = None
        
    def preprocess_data(self):
        """Prétraiter les données pour l'ACP: remplir les valeurs manquantes et standardiser."""
        # Imputation simple des valeurs manquantes par la moyenne des facteurs
        factors_filled = self.factors_df.fillna(self.factors_df.mean())
        
        # Standardiser les facteurs (moyenne=0, écart-type=1)
        self.scaler = StandardScaler()
        self.factors_scaled = pd.DataFrame(
            self.scaler.fit_transform(factors_filled),
            index=factors_filled.index,
            columns=factors_filled.columns
        )
        
        print(f"Données prétraitées: {self.factors_scaled.shape[0]} périodes, {self.factors_scaled.shape[1]} facteurs")
        
        return self.factors_scaled
    
    def run_pca(self, n_components=None, variance_threshold=0.95):
        """
        Exécute l'ACP sur les facteurs.
        
        Args:
            n_components (int, optional): Nombre de composantes à extraire
            variance_threshold (float, optional): Proportion de variance expliquée cible
            
        Returns:
            dict: Résultats de l'ACP
        """
        if not hasattr(self, 'factors_scaled'):
            self.preprocess_data()
            
        # Définir le nombre de composantes
        if n_components is None:
            # Utiliser le seuil de variance expliquée
            self.pca = PCA(n_components=variance_threshold, svd_solver='full')
        else:
            # Utiliser un nombre fixe de composantes
            self.pca = PCA(n_components=n_components, svd_solver='full')
            
        # Exécuter l'ACP
        principal_components = self.pca.fit_transform(self.factors_scaled)
        
        # Stocker les résultats
        self.explained_variance = self.pca.explained_variance_ratio_
        self.cum_explained_variance = np.cumsum(self.explained_variance)
        
        # Créer un DataFrame pour les composantes principales
        self.principal_factors = pd.DataFrame(
            principal_components,
            index=self.factors_scaled.index,
            columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
        )
        
        # Matrice des loadings (coefficients)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.factors_scaled.columns,
            columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
        )
        
        # Résumé des résultats
        print(f"ACP réalisée avec {principal_components.shape[1]} composantes principales")
        print(f"Variance expliquée totale: {self.cum_explained_variance[-1]:.4f}")
        
        # Créer un dictionnaire de résultats
        self.pca_results = {
            'n_components': principal_components.shape[1],
            'explained_variance_ratio': self.explained_variance,
            'cum_explained_variance': self.cum_explained_variance,
            'loadings': self.loadings,
            'principal_factors': self.principal_factors
        }
        
        return self.pca_results
    def run_factor_analysis_varimax(self, n_components=5):

        if not hasattr(self, 'factors_scaled'):
            self.preprocess_data()

    # Analyse Factorielle
        fa = FactorAnalysis(n_components=n_components)
        factors_fa = fa.fit_transform(self.factors_scaled)

    # Rotation Varimax
        rotator = Rotator(method='varimax')
        loadings_rotated = rotator.fit_transform(fa.components_.T)

    # Composantes rotées
        self.principal_factors = pd.DataFrame(
            factors_fa,
            index=self.factors_scaled.index,
            columns=[f'Factor{i+1}' for i in range(n_components)]
        )

    # Stocker les loadings rotés
        self.loadings = pd.DataFrame(
            loadings_rotated,
            index=self.factors_scaled.columns,
            columns=[f'Factor{i+1}' for i in range(n_components)]
        )

    # Calcul de la variance expliquée approximative
        explained_var = np.var(factors_fa, axis=0)
        self.explained_variance = explained_var / np.sum(explained_var)
        self.cum_explained_variance = np.cumsum(self.explained_variance)

        print(f"Analyse factorielle Varimax réalisée avec {n_components} facteurs")
        print(f"Variance expliquée totale: {self.cum_explained_variance[-1]:.4f}")

        return {
            'n_components': n_components,
            'explained_variance_ratio': self.explained_variance,
            'cum_explained_variance': self.cum_explained_variance,
            'loadings': self.loadings,
            'principal_factors': self.principal_factors
        }

    def purify_factors(self, n_components=5):
        """
        Purifier les facteurs originaux en retirant les expositions communes.
        
        Args:
            n_components (int): Nombre de composantes principales à utiliser pour la purification
            
        Returns:
            pd.DataFrame: Les facteurs purifiés
        """
        if self.pca_results is None or self.pca_results['n_components'] < n_components:
            self.run_pca(n_components=n_components)
        
        # Pour chaque facteur original, régresser contre les composantes principales
        purified_factors = {}
        
        for factor in self.factor_names:
            # Données du facteur original
            y = self.factors_df[factor]
            
            # Sélectionner les premières n_components principales
            X = self.principal_factors.iloc[:, :n_components]
            
            # Aligner les indices
            valid_idx = ~(y.isna() | X.isna().any(axis=1))
            
            if valid_idx.sum() < 30:  # Au moins 30 observations
                purified_factors[factor] = y
                continue
                
            # Ajouter une constante pour le terme d'intersection
            X_with_const = sm.add_constant(X[valid_idx])
            
            # Régresser le facteur contre les composantes principales
            model = sm.OLS(y[valid_idx], X_with_const).fit()
            
            # Les résidus représentent le facteur "purifié"
            residuals = pd.Series(index=y.index)
            residuals[valid_idx] = model.resid
            
            # Stocker le facteur purifié
            purified_factors[factor] = residuals
        
        # Créer un DataFrame avec les facteurs purifiés
        self.purified_factors = pd.DataFrame(purified_factors)
        
        print(f"Facteurs purifiés créés : {self.purified_factors.shape[1]} facteurs")
        
        return self.purified_factors
    
    def create_factor_clusters_hierarchical(self, method='ward', n_clusters=None, threshold=None):
        """
        Crée des clusters de facteurs basés sur leur corrélation.
        
        Args:
            method (str): Méthode de linkage ('ward', 'complete', 'average', etc.)
            n_clusters (int, optional): Nombre de clusters à former
            threshold (float, optional): Seuil de distance pour former les clusters
            
        Returns:
            dict: Dictionnaire associant chaque facteur à son cluster
        """
        # Calculer la matrice de corrélation
        corr_matrix = self.factors_df.corr()
        
        # Convertir en matrice de distance (1 - abs(corr))
        dist_matrix = 1 - corr_matrix.abs()
        
        # Appliquer la classification hiérarchique
        Z = linkage(dist_matrix.values, method=method)
        
        # Déterminer les clusters
        if n_clusters is not None:
            labels = fcluster(Z, n_clusters, criterion='maxclust')
        elif threshold is not None:
            labels = fcluster(Z, threshold, criterion='distance')
        else:
            # Par défaut, utiliser 12 clusters comme dans l'article original
            labels = fcluster(Z, 12, criterion='maxclust')
        
        # Créer un dictionnaire associant chaque facteur à son cluster
        clusters = {}
        for i, factor in enumerate(self.factor_names):
            clusters[factor] = f'Cluster_{labels[i]}'
        
        # Stocker les résultats
        self.factor_clusters = clusters
        self.linkage_matrix = Z
        
        print(f"Clusters de facteurs créés : {len(set(clusters.values()))} clusters")
        
        return clusters, Z
    
    def plot_scree(self, n_components=20):
        """Trace le diagramme des éboulis (Scree plot)."""
        if self.pca_results is None:
            self.run_pca(n_components=n_components)
            
        n_comps = min(n_components, len(self.explained_variance))
            
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Axe principal pour la variance expliquée individuelle
        ax1.bar(range(1, n_comps+1), self.explained_variance[:n_comps], 
                alpha=0.7, color='steelblue', label='Variance expliquée')
        ax1.set_xlabel('Composante principale')
        ax1.set_ylabel('Variance expliquée (ratio)')
        ax1.set_xticks(range(1, n_comps+1))
        
        # Axe secondaire pour la variance expliquée cumulée
        ax2 = ax1.twinx()
        ax2.plot(range(1, n_comps+1), self.cum_explained_variance[:n_comps], 
                 'o-', color='red', label='Variance cumulée')
        ax2.axhline(y=0.9, linestyle='--', color='darkred', alpha=0.7)
        ax2.set_ylabel('Variance expliquée cumulée')
        ax2.set_ylim(0, 1.05)
        
        # Légende
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Diagramme des éboulis (Scree Plot)')
        plt.tight_layout()
        
        return fig
    
    def plot_factor_loadings(self, n_components=5, n_top_factors=15):
        """Trace les coefficients (loadings) des principaux facteurs."""
        if self.pca_results is None:
            self.run_pca(n_components=n_components)
            
        # Limiter aux n_components premières composantes
        n_comps = min(n_components, len(self.explained_variance))
        loadings_subset = self.loadings.iloc[:, :n_comps]
        
        # Pour chaque composante, sélectionner les n_top_factors avec les loadings les plus forts
        fig, axes = plt.subplots(1, n_comps, figsize=(15, 8), sharey=True)
        
        for i in range(n_comps):
            # Trier les loadings par valeur absolue
            sorted_loadings = loadings_subset.iloc[:, i].abs().sort_values(ascending=False)
            top_factors = sorted_loadings.index[:n_top_factors]
            
            # Tracer les loadings pour les facteurs principaux
            bars = axes[i].barh(
                range(len(top_factors)), 
                loadings_subset.loc[top_factors, f'PC{i+1}'],
                color=[plt.cm.RdBu(0.9 if x >= 0 else 0.1) for x in loadings_subset.loc[top_factors, f'PC{i+1}']]
            )
            
            # Ajouter les noms des facteurs
            axes[i].set_yticks(range(len(top_factors)))
            axes[i].set_yticklabels(top_factors)
            
            # Ajouter une ligne verticale à zéro
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Titre pour chaque composante
            var_explained = self.explained_variance[i] * 100
            axes[i].set_title(f'PC{i+1}\n({var_explained:.1f}%)')
            
            # Limiter l'axe x pour améliorer la lisibilité
            max_val = loadings_subset.iloc[:, i].abs().max() * 1.2
            axes[i].set_xlim(-max_val, max_val)
            
        # Ajustements globaux
        fig.suptitle('Coefficients des principales composantes', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig
    
    def plot_factor_dendrogram(self, figsize=(12, 8)):
        """Trace le dendrogramme des facteurs basé sur la classification hiérarchique."""
        if not hasattr(self, 'linkage_matrix'):
            self.create_factor_clusters_hierarchical()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer le dendrogramme
        dendrogram(
            self.linkage_matrix,
            labels=self.factor_names,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax
        )
        
        # Titre
        plt.title('Classification hiérarchique des facteurs', fontsize=14)
        plt.xlabel('Facteurs')
        plt.ylabel('Distance')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_heatmap(self, n_top_factors=30, purified=False):
        """Trace une heatmap de corrélation entre les principaux facteurs."""
        # Choisir les données à utiliser
        if purified and self.purified_factors is not None:
            data = self.purified_factors
            title_prefix = "Facteurs purifiés"
        else:
            data = self.factors_df
            title_prefix = "Facteurs originaux"
            
        # Calculer la matrice de corrélation
        corr_matrix = data.corr()
        
        # Sélectionner les facteurs avec la variance expliquée la plus élevée
        if len(self.factor_names) > n_top_factors:
            # Calculer la variance de chaque facteur
            variances = data.var()
            top_factors = variances.sort_values(ascending=False).index[:n_top_factors]
            corr_subset = corr_matrix.loc[top_factors, top_factors]
        else:
            corr_subset = corr_matrix
            
        # Tracer la heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Utiliser une palette divergente
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Tracer la heatmap
        sns.heatmap(
            corr_subset, 
            ax=ax,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )
        
        # Titre
        plt.title(f'Matrice de corrélation ({title_prefix})', fontsize=14)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def compare_original_vs_purified(self, factors_to_show=5, n_components=5):
        """
        Compare les facteurs originaux et purifiés en termes de corrélation.
        
        Args:
            factors_to_show (int): Nombre de facteurs à afficher
            n_components (int): Nombre de composantes principales utilisées pour la purification
            
        Returns:
            matplotlib.figure.Figure: La figure créée
        """
        if self.purified_factors is None:
            self.purify_factors(n_components=n_components)
            
        # Créer une figure avec deux grilles
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
        
        # Sélectionner les facteurs avec la plus forte corrélation entre eux
        corr_matrix = self.factors_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Ignorer l'autodiagnostic
        
        # Calculer la corrélation moyenne pour chaque facteur
        mean_corr = corr_matrix.mean()
        top_factors = mean_corr.sort_values(ascending=False).index[:factors_to_show]
        
        # 1. Matrice de corrélation des facteurs originaux
        ax1 = fig.add_subplot(gs[0, 0])
        orig_corr = self.factors_df[top_factors].corr()
        sns.heatmap(orig_corr, ax=ax1, cmap='RdBu_r', vmin=-1, vmax=1, 
                    annot=True, fmt='.2f', linewidths=.5, cbar=False)
        ax1.set_title('Corrélation des facteurs originaux')
        
        # 2. Matrice de corrélation des facteurs purifiés
        ax2 = fig.add_subplot(gs[1, 0])
        purified_corr = self.purified_factors[top_factors].corr()
        sns.heatmap(purified_corr, ax=ax2, cmap='RdBu_r', vmin=-1, vmax=1, 
                    annot=True, fmt='.2f', linewidths=.5, cbar=False)
        ax2.set_title('Corrélation des facteurs purifiés')
        
        # 3. Barplot des corrélations moyennes
        ax3 = fig.add_subplot(gs[:, 1])
        
        # Calculer les corrélations moyennes absolues pour chaque facteur
        mean_orig_corr = orig_corr.abs().mean().drop_duplicates()
        mean_purified_corr = purified_corr.abs().mean().drop_duplicates()
        
        # Créer un DataFrame pour le barplot
        comparison_df = pd.DataFrame({
            'Original': mean_orig_corr,
            'Purifié': mean_purified_corr
        })
        
        # Trier par corrélation originale
        comparison_df = comparison_df.sort_values('Original', ascending=False)
        
        # Tracer le barplot
        comparison_df.plot(kind='barh', ax=ax3, alpha=0.7)
        ax3.set_title('Corrélation moyenne absolue')
        ax3.set_xlabel('Corrélation moyenne')
        ax3.grid(True, alpha=0.3)
        
        # Ajouter une légende
        ax3.legend(title='Type de facteur')
        
        plt.tight_layout()
        plt.suptitle(f'Comparaison des facteurs originaux vs purifiés (PCs={n_components})', 
                     fontsize=16, y=1.02)
        
        return fig
    
    def evaluate_purified_factors(self, market_return=None, n_components=5):
        """
        Évalue les performances des facteurs purifiés vs originaux.
        
        Args:
            market_return (pd.Series, optional): Rendement du marché
            n_components (int): Nombre de composantes principales pour la purification
            
        Returns:
            tuple: (DataFrame des résultats, Figure)
        """
        if self.purified_factors is None:
            self.purify_factors(n_components=n_components)
            
        # Si market_return n'est pas fourni, utiliser celui stocké dans la classe
        if market_return is None:
            if self.market_return is not None:
                market_return = self.market_return
            else:
                # Utiliser la moyenne comme proxy simple
                market_return = self.factors_df.mean(axis=1)
        
        # Résultats pour stocker les statistiques
        results = []
        
        # Évaluer les facteurs originaux
        for factor in self.factor_names:
            # Effectuer une régression contre le marché
            y = self.factors_df[factor]
            X = sm.add_constant(market_return)
            
            # Indices valides
            valid_idx = ~(y.isna() | market_return.isna())
            
            if valid_idx.sum() < 30:  # Au moins 30 observations
                continue
                
            model = sm.OLS(y[valid_idx], X[valid_idx]).fit()
            
            # Calculer l'information ratio
            ir = model.params[0] / model.resid.std()
            
            # Stocker les résultats
            results.append({
                'factor': factor,
                'type': 'Original',
                'alpha': model.params[0],
                't_stat': model.tvalues[0],
                'p_value': model.pvalues[0],
                'r_squared': model.rsquared,
                'sharpe': y[valid_idx].mean() / y[valid_idx].std(),
                'info_ratio': ir
            })
        
        # Évaluer les facteurs purifiés
        for factor in self.factor_names:
            # Effectuer une régression contre le marché
            y = self.purified_factors[factor]
            X = sm.add_constant(market_return)
            
            # Indices valides
            valid_idx = ~(y.isna() | market_return.isna())
            
            if valid_idx.sum() < 30:  # Au moins 30 observations
                continue
                
            model = sm.OLS(y[valid_idx], X[valid_idx]).fit()
            
            # Calculer l'information ratio
            ir = model.params[0] / model.resid.std()
            
            # Stocker les résultats
            results.append({
                'factor': factor,
                'type': 'Purifié',
                'alpha': model.params[0],
                't_stat': model.tvalues[0],
                'p_value': model.pvalues[0],
                'r_squared': model.rsquared,
                'sharpe': y[valid_idx].mean() / y[valid_idx].std(),
                'info_ratio': ir
            })
        
        # Créer un DataFrame
        results_df = pd.DataFrame(results)
        
        # Compter les facteurs significatifs
        orig_sig = results_df[(results_df['type'] == 'Original') & (results_df['t_stat'].abs() > 2)].shape[0]
        purif_sig = results_df[(results_df['type'] == 'Purifié') & (results_df['t_stat'].abs() > 2)].shape[0]
        
        print(f"Facteurs originaux significatifs (|t| > 2): {orig_sig}")
        print(f"Facteurs purifiés significatifs (|t| > 2): {purif_sig}")
        
        # Visualiser les résultats
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Alpha comparison
        sns.boxplot(x='type', y='alpha', data=results_df, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution des Alphas')
        axes[0, 0].set_ylabel('Alpha')
        axes[0, 0].set_xlabel('')
        
        # 2. t-statistic comparison
        sns.boxplot(x='type', y='t_stat', data=results_df, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution des t-statistics')
        axes[0, 1].set_ylabel('t-statistic')
        axes[0, 1].set_xlabel('')
        axes[0, 1].axhline(y=2, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        
        # 3. Information ratio comparison
        sns.boxplot(x='type', y='info_ratio', data=results_df, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution des Information Ratios')
        axes[1, 0].set_ylabel('Information Ratio')
        axes[1, 0].set_xlabel('')
        
        # 4. R-squared comparison
        sns.boxplot(x='type', y='r_squared', data=results_df, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution des R²')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_xlabel('')
        
        plt.tight_layout()
        plt.suptitle(f'Comparaison des performances: Facteurs originaux vs purifiés (PCs={n_components})', 
                     fontsize=16, y=1.02)
        
        return results_df, fig
    
if __name__ == "__main__":
    # Fermer toutes les figures matplotlib ouvertes
    plt.close('all')

    # Charger les données
    data_loader = DataLoader('VW_cap', '1993-08-01', '2021-12-31')
    factors_df, market_ret = data_loader.load_factor_data(region='world')

    # Convertir en décimal si nécessaire
    if abs(factors_df.mean().mean()) > 0.05:
        print("Conversion des données de pourcentage en décimal")
        factors_df = factors_df / 100
        market_ret = market_ret / 100

    # Initialiser et exécuter l'analyse PCA
    pca_extension = PCAPurification(factors_df, market_ret)

    # Étapes du PCA
    pca_extension.preprocess_data()
    pca_results = pca_extension.run_pca(n_components=10)

    # Purifier les facteurs
    purified_factors = pca_extension.purify_factors(n_components=10)

    # Afficher les résultats
    print("\nRésultats de l'ACP:")
    print(f"Nombre de composantes: {pca_results['n_components']}")
    print(f"Variance expliquée totale: {pca_results['cum_explained_variance'][-1]:.4f}")

    # Créer un dossier pour les résultats
    output_dir = "../../Factor Zoo Zip.extensions/results_pca"
    os.makedirs(output_dir, exist_ok=True)

    # Dictionnaire des figures à sauvegarder
    figures_to_save = {
        'scree_plot': pca_extension.plot_scree(n_components=10),
        'loadings': pca_extension.plot_factor_loadings(n_components=5),
        'dendrogram': pca_extension.plot_factor_dendrogram(),
        'comparison': pca_extension.compare_original_vs_purified(),
    }

    # Évaluation des facteurs purifiés
    results_df, fig_eval = pca_extension.evaluate_purified_factors(market_return=market_ret)
    figures_to_save['evaluation'] = fig_eval

    # Sauvegarder toutes les figures
    for name, fig in figures_to_save.items():
        filename = os.path.join(output_dir, f'{name}.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fermer la figure pour libérer la mémoire
        print(f"Sauvegardé: {filename}")

    print(f"\nAnalyse PCA terminée! Tous les graphiques ont été sauvegardés dans '{output_dir}'.")
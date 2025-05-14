# ml_factor_selection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MLFactorSelection:
    """
    Extensions Machine Learning pour la sélection de facteurs
    Complète la méthode itérative classique avec des approches ML
    """
    
    def __init__(self, factors_df, market_return, target_returns=None):
        """
        Initialise avec les données de facteurs
        
        Args:
            factors_df: DataFrame avec les retours des facteurs
            market_return: Série des retours du marché
            target_returns: Retours cibles (par défaut: moyenne cross-sectionnelle)
        """
        self.factors_df = factors_df.copy()
        self.market_return = market_return.copy()
        
        # Si pas de retours cibles, utiliser la moyenne des facteurs
        if target_returns is None:
            self.target_returns = factors_df.mean(axis=1)
        else:
            self.target_returns = target_returns
        
        # Aligner les indices
        common_index = factors_df.index.intersection(market_return.index)
        self.factors_df = self.factors_df.loc[common_index]
        self.market_return = self.market_return.loc[common_index]
        self.target_returns = self.target_returns.loc[common_index]
        
        # Scaler pour normalisation
        self.scaler = StandardScaler()
    
    def prepare_data(self, include_market=True):
        """Prépare les données pour le ML"""
        X = self.factors_df.copy()
        
        if include_market:
            X['market'] = self.market_return
        
        # Gérer les valeurs manquantes
        X = X.fillna(X.mean())
        
        # Normaliser
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        return X_scaled, self.target_returns
    
    def random_forest_importance(self, n_estimators=100, max_depth=10):
        """
        Utilise Random Forest pour mesurer l'importance des facteurs
        
        Returns:
            DataFrame avec l'importance de chaque facteur
        """
        X, y = self.prepare_data()
        
        # Entraîner le modèle
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        rf.fit(X, y)
        
        # Extraire l'importance
        importance = pd.DataFrame({
            'factor': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculer aussi l'importance par permutation
        perm_importance = []
        base_score = rf.score(X, y)
        
        for col in X.columns:
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col])
            perm_score = rf.score(X_perm, y)
            perm_importance.append(base_score - perm_score)
        
        importance['permutation_importance'] = perm_importance
        
        return importance
    
    def lasso_selection(self, alphas=None, cv=5):
        """
        Sélection de facteurs par LASSO avec validation croisée
        
        Args:
            alphas: Grille de paramètres alpha à tester
            cv: Nombre de folds pour la validation croisée
            
        Returns:
            Dictionnaire avec les facteurs sélectionnés et leurs coefficients
        """
        X, y = self.prepare_data()
        
        if alphas is None:
            alphas = np.logspace(-4, 1, 50)
        
        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Trouver le meilleur alpha par CV
        best_alpha = None
        best_score = -np.inf
        
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, random_state=42)
            scores = cross_val_score(lasso, X, y, cv=tscv, scoring='r2')
            
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_alpha = alpha
        
        # Entraîner avec le meilleur alpha
        lasso_best = Lasso(alpha=best_alpha, random_state=42)
        lasso_best.fit(X, y)
        
        # Extraire les facteurs sélectionnés
        selected_factors = pd.DataFrame({
            'factor': X.columns,
            'coefficient': lasso_best.coef_
        })
        
        selected_factors = selected_factors[selected_factors['coefficient'] != 0]
        selected_factors['abs_coefficient'] = selected_factors['coefficient'].abs()
        selected_factors = selected_factors.sort_values('abs_coefficient', ascending=False)
        
        return {
            'selected_factors': selected_factors,
            'best_alpha': best_alpha,
            'r2_score': best_score,
            'model': lasso_best
        }
    
    def elastic_net_selection(self, l1_ratios=None, alphas=None, cv=5):
        """
        Sélection par Elastic Net (combinaison LASSO + Ridge)
        """
        X, y = self.prepare_data()
        
        if l1_ratios is None:
            l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
        
        if alphas is None:
            alphas = np.logspace(-4, 1, 20)
        
        tscv = TimeSeriesSplit(n_splits=cv)
        
        best_params = {'alpha': None, 'l1_ratio': None}
        best_score = -np.inf
        
        # Grid search
        for l1_ratio in l1_ratios:
            for alpha in alphas:
                en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                scores = cross_val_score(en, X, y, cv=tscv, scoring='r2')
                
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        
        # Entraîner avec les meilleurs paramètres
        en_best = ElasticNet(**best_params, random_state=42)
        en_best.fit(X, y)
        
        # Extraire les facteurs sélectionnés
        selected_factors = pd.DataFrame({
            'factor': X.columns,
            'coefficient': en_best.coef_
        })
        
        selected_factors = selected_factors[selected_factors['coefficient'] != 0]
        selected_factors['abs_coefficient'] = selected_factors['coefficient'].abs()
        selected_factors = selected_factors.sort_values('abs_coefficient', ascending=False)
        
        return {
            'selected_factors': selected_factors,
            'best_params': best_params,
            'r2_score': best_score,
            'model': en_best
        }
    
    def gradient_boosting_importance(self, n_estimators=100, max_depth=3):
        """
        Utilise Gradient Boosting pour l'importance des facteurs
        Capture les non-linéarités
        """
        X, y = self.prepare_data()
        
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        gb.fit(X, y)
        
        importance = pd.DataFrame({
            'factor': X.columns,
            'importance': gb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def mutual_information_selection(self, k_features=10):
        """
        Sélection basée sur l'information mutuelle
        Capture les relations non-linéaires
        """
        X, y = self.prepare_data()
        
        # Calculer l'information mutuelle
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'factor': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Sélectionner les k meilleurs
        selector = SelectKBest(mutual_info_regression, k=k_features)
        selector.fit(X, y)
        
        selected_mask = selector.get_support()
        selected_factors = X.columns[selected_mask].tolist()
        
        return {
            'all_scores': mi_df,
            'selected_factors': selected_factors,
            'selector': selector
        }
    
    def ensemble_selection(self, methods=['lasso', 'rf', 'gb'], top_k=10):
        """
        Combine plusieurs méthodes de sélection
        
        Args:
            methods: Liste des méthodes à utiliser
            top_k: Nombre de facteurs à sélectionner par méthode
        """
        results = {}
        all_rankings = []
        
        # Random Forest
        if 'rf' in methods:
            rf_imp = self.random_forest_importance()
            rf_top = rf_imp.head(top_k)['factor'].tolist()
            results['rf'] = rf_top
            all_rankings.append(rf_imp.reset_index()[['factor', 'importance']])
        
        # LASSO
        if 'lasso' in methods:
            lasso_results = self.lasso_selection()
            lasso_top = lasso_results['selected_factors'].head(top_k)['factor'].tolist()
            results['lasso'] = lasso_top
            
            lasso_rank = lasso_results['selected_factors'][['factor', 'abs_coefficient']]
            lasso_rank.columns = ['factor', 'importance']
            all_rankings.append(lasso_rank)
        
        # Elastic Net
        if 'elastic_net' in methods:
            en_results = self.elastic_net_selection()
            en_top = en_results['selected_factors'].head(top_k)['factor'].tolist()
            results['elastic_net'] = en_top
            
            en_rank = en_results['selected_factors'][['factor', 'abs_coefficient']]
            en_rank.columns = ['factor', 'importance']
            all_rankings.append(en_rank)
        
        # Gradient Boosting
        if 'gb' in methods:
            gb_imp = self.gradient_boosting_importance()
            gb_top = gb_imp.head(top_k)['factor'].tolist()
            results['gb'] = gb_top
            all_rankings.append(gb_imp)
        
        # Mutual Information
        if 'mi' in methods:
            mi_results = self.mutual_information_selection(k_features=top_k)
            results['mi'] = mi_results['selected_factors']
            
            mi_rank = mi_results['all_scores'][['factor', 'mi_score']]
            mi_rank.columns = ['factor', 'importance']
            all_rankings.append(mi_rank)
        
        # Calculer le consensus
        all_factors = []
        for method_factors in results.values():
            all_factors.extend(method_factors)
        
        factor_counts = pd.Series(all_factors).value_counts()
        
        # Score agrégé
        aggregated_scores = {}
        for ranking in all_rankings:
            # Normaliser les scores entre 0 et 1
            ranking = ranking.copy()
            ranking['normalized_score'] = (
                ranking['importance'] / ranking['importance'].max()
            )
            
            for _, row in ranking.iterrows():
                factor = row['factor']
                score = row['normalized_score']
                
                if factor not in aggregated_scores:
                    aggregated_scores[factor] = []
                aggregated_scores[factor].append(score)
        
        # Calculer le score moyen
        consensus_ranking = pd.DataFrame([
            {
                'factor': factor,
                'mean_score': np.mean(scores),
                'count': factor_counts.get(factor, 0),
                'methods': len(scores)
            }
            for factor, scores in aggregated_scores.items()
        ])
        
        consensus_ranking['combined_score'] = (
            consensus_ranking['mean_score'] * consensus_ranking['count'] / len(methods)
        )
        
        consensus_ranking = consensus_ranking.sort_values(
            'combined_score', ascending=False
        )
        
        return {
            'individual_results': results,
            'consensus_ranking': consensus_ranking,
            'factor_counts': factor_counts
        }
    
    def visualize_results(self, results_dict):
        """Visualise les résultats de sélection"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Importance par méthode
        ax = axes[0, 0]
        importance_data = []
        
        if 'rf' in results_dict.get('individual_results', {}):
            rf_imp = self.random_forest_importance()
            importance_data.append(
                rf_imp.head(10).assign(method='Random Forest')
            )
        
        if 'gb' in results_dict.get('individual_results', {}):
            gb_imp = self.gradient_boosting_importance()
            importance_data.append(
                gb_imp.head(10).assign(method='Gradient Boosting')
            )
        
        if importance_data:
            combined_imp = pd.concat(importance_data)
            sns.barplot(
                data=combined_imp,
                x='importance',
                y='factor',
                hue='method',
                ax=ax
            )
            ax.set_title('Feature Importance par Méthode')
        
        # 2. Consensus
        ax = axes[0, 1]
        if 'consensus_ranking' in results_dict:
            consensus = results_dict['consensus_ranking'].head(15)
            sns.barplot(
                x='combined_score',
                y='factor',
                data=consensus,
                ax=ax
            )
            ax.set_title('Score Consensus')
        
        # 3. Nombre de méthodes sélectionnant chaque facteur
        ax = axes[1, 0]
        if 'factor_counts' in results_dict:
            counts = results_dict['factor_counts'].head(15)
            counts.plot(kind='barh', ax=ax)
            ax.set_title('Nombre de méthodes sélectionnant le facteur')
            ax.set_xlabel('Count')
        
        # 4. Matrice de corrélation des facteurs sélectionnés
        ax = axes[1, 1]
        if 'consensus_ranking' in results_dict:
            top_factors = results_dict['consensus_ranking'].head(10)['factor'].tolist()
            if len(top_factors) > 1:
                corr_matrix = self.factors_df[top_factors].corr()
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    ax=ax,
                    fmt='.2f'
                )
                ax.set_title('Corrélation des Top Facteurs')
        
        plt.tight_layout()
        return fig
    
    def forward_selection(self, max_features=15, scoring='r2'):
        """
        Sélection forward avec évaluation out-of-sample
        """
        X, y = self.prepare_data()
        
        selected_features = []
        remaining_features = list(X.columns)
        scores = []
        
        tscv = TimeSeriesSplit(n_splits=5)
        base_model = Ridge(alpha=1.0)
        
        for i in range(max_features):
            if not remaining_features:
                break
                
            best_score = -np.inf
            best_feature = None
            
            # Tester chaque feature restante
            for feature in remaining_features:
                test_features = selected_features + [feature]
                X_test = X[test_features]
                
                # Cross-validation temporelle
                cv_scores = cross_val_score(
                    base_model, X_test, y, cv=tscv, scoring=scoring
                )
                score = cv_scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            # Ajouter la meilleure feature
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                scores.append(best_score)
                
                print(f"Step {i+1}: Added {best_feature}, Score: {best_score:.4f}")
            else:
                break
        
        # Entraîner le modèle final
        final_model = Ridge(alpha=1.0)
        final_model.fit(X[selected_features], y)
        
        return {
            'selected_features': selected_features,
            'scores': scores,
            'final_model': final_model
        }
    
    def compare_with_traditional(self, traditional_factors, ml_factors):
        """
        Compare les facteurs ML avec la sélection traditionnelle
        """
        X, y = self.prepare_data()
        
        results = {}
        models = {}
        
        # Évaluer les facteurs traditionnels
        if traditional_factors:
            X_trad = X[traditional_factors]
            model_trad = Ridge(alpha=1.0)
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores_trad = cross_val_score(
                model_trad, X_trad, y, cv=tscv, scoring='r2'
            )
            
            model_trad.fit(X_trad, y)
            results['traditional'] = {
                'r2_cv': scores_trad.mean(),
                'r2_std': scores_trad.std(),
                'n_factors': len(traditional_factors)
            }
            models['traditional'] = model_trad
        
        # Évaluer les facteurs ML
        if ml_factors:
            X_ml = X[ml_factors]
            model_ml = Ridge(alpha=1.0)
            
            scores_ml = cross_val_score(
                model_ml, X_ml, y, cv=tscv, scoring='r2'
            )
            
            model_ml.fit(X_ml, y)
            results['ml'] = {
                'r2_cv': scores_ml.mean(),
                'r2_std': scores_ml.std(),
                'n_factors': len(ml_factors)
            }
            models['ml'] = model_ml
        
        # Facteurs communs
        common_factors = list(set(traditional_factors) & set(ml_factors))
        results['common_factors'] = common_factors
        results['n_common'] = len(common_factors)
        
        return results, models

############################# example #################################
from data_loader import DataLoader

# Charger vos données Factor Zoo
data_loader = DataLoader('VW_cap', '1993-08-01', '2021-12-31')
factors_df, market_ret = data_loader.load_factor_data('US')

# Convertir en décimal si nécessaire
if abs(market_ret.mean()) > 0.05:
    factors_df = factors_df / 100
    market_ret = market_ret / 100

# Utiliser ML Factor Selection
ml_selector = MLFactorSelection(factors_df, market_ret)

# Sélection ensemble
ensemble_results = ml_selector.ensemble_selection(
    methods=['lasso', 'rf', 'gb', 'mi', 'elastic_net'],
    top_k=15
)

# Afficher les top 15 facteurs
print("Top 15 facteurs selon l'approche ML:")
print(ensemble_results['consensus_ranking'].head(15))

# Comparer avec la sélection traditionnelle
# (Remplacez par vos vrais facteurs traditionnels)
traditional_factors = ['cop_at', 'noa_gr1a', 'saleq_gr1', 'ival_me', 'resff3_12_1']
ml_top_factors = ensemble_results['consensus_ranking'].head(5)['factor'].tolist()

comparison, _ = ml_selector.compare_with_traditional(
    traditional_factors,
    ml_top_factors
)

print("\nComparaison Traditionnel vs ML:")
print(f"R² Traditionnel: {comparison['traditional']['r2_cv']:.4f}")
print(f"R² ML: {comparison['ml']['r2_cv']:.4f}")
print(f"Amélioration: {(comparison['ml']['r2_cv'] - comparison['traditional']['r2_cv'])*100:.1f}%")
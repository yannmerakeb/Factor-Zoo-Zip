# test_factor_selection.py
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
import warnings
warnings.filterwarnings('ignore')


class TestFactorSelection(unittest.TestCase):
    """Tests unitaires pour la sélection de facteurs"""
    
    def setUp(self):
        """Configuration des données de test"""
        np.random.seed(42)
        self.n_periods = 120
        self.n_factors = 10
        
        # Générer des données synthétiques
        self.dates = pd.date_range('2010-01-01', periods=self.n_periods, freq='ME')
        
        # Marché avec tendance
        self.market_return = pd.Series(
            np.random.normal(0.01, 0.02, self.n_periods), 
            index=self.dates
        )
        
        # Facteurs avec différents niveaux d'alpha
        self.factors_data = {}
        alphas = [0.005, 0.003, 0.002, 0.001, 0.0005, 0, 0, 0, 0, 0]
        
        for i in range(self.n_factors):
            factor_returns = (
                alphas[i] + 
                0.5 * self.market_return + 
                np.random.normal(0, 0.015, self.n_periods)
            )
            self.factors_data[f'factor_{i}'] = factor_returns
        
        self.factors_df = pd.DataFrame(self.factors_data, index=self.dates)
    
    def test_regression_basic(self):
        """Test basique de régression"""
        y = self.factors_df.iloc[:, 0]
        X = self.market_return
        
        # Effectuer la régression
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const, missing='drop')
        res = model.fit()
        
        # Vérifier les propriétés de base
        self.assertTrue(hasattr(res, 'params'))
        self.assertTrue(hasattr(res, 'tvalues'))
        self.assertTrue(hasattr(res, 'pvalues'))
        
        # Vérifier que l'alpha est positif (car on l'a généré positif)
        self.assertGreater(res.params.iloc[0], 0)
    
    def test_grs_calculation(self):
        """Test du calcul GRS de base"""
        # Simuler des alphas et résidus
        n_factors = 5
        T = 100
        K = 2
        
        alphas = np.random.normal(0, 0.001, n_factors)
        residuals = np.random.normal(0, 0.01, (T, n_factors))
        factors = np.random.normal(0, 0.02, (T, K))
        
        # Calcul GRS simplifié
        try:
            Sigma = np.cov(residuals.T)
            Omega = np.cov(factors.T)
            f_bar = np.mean(factors, axis=0)
            
            # Vérifier que les matrices sont calculables
            self.assertEqual(Sigma.shape, (n_factors, n_factors))
            self.assertEqual(Omega.shape, (K, K))
            self.assertEqual(f_bar.shape, (K,))
            
            # Vérifier que les matrices sont inversibles
            det_sigma = np.linalg.det(Sigma)
            det_omega = np.linalg.det(Omega)
            
            self.assertNotEqual(det_sigma, 0)
            self.assertNotEqual(det_omega, 0)
            
        except Exception as e:
            self.fail(f"Erreur dans le calcul GRS: {e}")
    
    def test_iterative_selection_logic(self):
        """Test de la logique de sélection itérative"""
        available_factors = list(self.factors_df.columns)
        selected_factors = []
        
        # Premier pas: juste le marché comme base
        X_base = self.market_return.to_frame('market')
        
        # Tester chaque facteur
        t_stats = {}
        for factor in available_factors:
            y = self.factors_df[factor]
            
            # Régression
            X_with_const = sm.add_constant(X_base)
            model = sm.OLS(y, X_with_const, missing='drop')
            res = model.fit()
            
            t_stats[factor] = abs(res.tvalues.iloc[0])
        
        # Le meilleur facteur devrait être celui avec le plus grand alpha
        best_factor = max(t_stats, key=t_stats.get)
        
        # Vérifier que le meilleur facteur est l'un des premiers
        # (qui ont les plus grands alphas dans notre setup)
        self.assertIn(best_factor, ['factor_0', 'factor_1', 'factor_2', 'factor_3', 'factor_4'])
    
    def test_with_missing_values(self):
        """Test avec valeurs manquantes"""
        # Données avec NaN
        factor_data = np.random.normal(0.001, 0.02, 100)
        factor_data[10:20] = np.nan  # Introduire des NaN
        
        y = pd.Series(factor_data)
        X = pd.Series(np.random.normal(0.01, 0.02, 100))
        
        # La régression devrait gérer les valeurs manquantes
        X_with_const = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X_with_const, missing='drop')
            res = model.fit()
            
            # Vérifier que les résultats sont valides
            self.assertTrue(np.isfinite(res.params.iloc[0]))
            self.assertTrue(np.isfinite(res.tvalues.iloc[0]))
            
        except Exception as e:
            self.fail(f"La régression devrait gérer les NaN: {e}")


class TestDataPreparation(unittest.TestCase):
    """Tests pour la préparation des données"""
    
    def test_data_alignment(self):
        """Test de l'alignement des indices temporels"""
        # Créer des données avec des indices non alignés
        dates1 = pd.date_range('2010-01-01', periods=100, freq='ME')
        dates2 = pd.date_range('2010-02-01', periods=100, freq='ME')
        
        factors = pd.DataFrame({
            'factor_1': np.random.normal(0, 0.02, 100)
        }, index=dates1)
        
        market = pd.Series(
            np.random.normal(0, 0.02, 100),
            index=dates2
        )
        
        # Aligner les données
        common_index = factors.index.intersection(market.index)
        
        self.assertTrue(len(common_index) > 0)
        self.assertEqual(len(common_index), 99)  # 1 mois de différence
    
    def test_normalization(self):
        """Test de la normalisation des données"""
        # Données à différentes échelles
        factors = pd.DataFrame({
            'factor_1': np.random.normal(0, 0.02, 100),  # Petite échelle
            'factor_2': np.random.normal(0, 2.0, 100),   # Grande échelle
        })
        
        # Normaliser
        normalized = (factors - factors.mean()) / factors.std()
        
        # Vérifier que les moyennes sont proches de 0
        self.assertAlmostEqual(normalized['factor_1'].mean(), 0, places=10)
        self.assertAlmostEqual(normalized['factor_2'].mean(), 0, places=10)
        
        # Vérifier que les écarts-types sont proches de 1
        self.assertAlmostEqual(normalized['factor_1'].std(), 1, places=10)
        self.assertAlmostEqual(normalized['factor_2'].std(), 1, places=10)


if __name__ == '__main__':
    unittest.main()
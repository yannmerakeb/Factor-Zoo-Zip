# ExtensionTemporelleAvancee.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import minimal pour éviter les dépendances
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    MARKOV_AVAILABLE = True
except ImportError:
    MARKOV_AVAILABLE = False
    print("Warning: Markov switching models not available. Install with: pip install statsmodels>=0.13.0")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: Ruptures not available. Install with: pip install ruptures")

from factor_zoo import IterativeFactorSelection

class TemporalDynamicsAnalysis:
    """
    Extension du Factor Zoo qui analyse les dynamiques temporelles des facteurs.
    """
    
    def __init__(self, factors_df: pd.DataFrame, market_return: pd.Series):
        self.factors_df = factors_df.copy()
        self.market_return = market_return.copy()
        self.n_factors = factors_df.shape[1]
        self.factor_names = factors_df.columns.tolist()
        
        self.breakpoints = None
        self.regime_models = None
        self.adaptive_selection = None
        self.temporal_stability = None
    
    def adaptive_factor_selection(self, window_size: int = 60, step_size: int = 12) -> pd.DataFrame:
        """
        Sélection adaptative de facteurs sur fenêtres glissantes.
        """
        print(f"Sélection adaptative (fenêtre={window_size}, pas={step_size})...")
        
        results = []
        
        for start in range(0, len(self.factors_df) - window_size + 1, step_size):
            end = start + window_size
            
            # Données de la fenêtre
            window_factors = self.factors_df.iloc[start:end]
            window_market = self.market_return.iloc[start:end]
            
            # Date de fin de la fenêtre
            window_date = self.factors_df.index[end-1]
            
            # Sélection itérative
            selector = IterativeFactorSelection(
                window_factors, 
                window_market,
                significance_threshold=3.0
            )
            selection = selector.select_factors(max_factors=15)
            
            # Stocker les résultats
            window_result = {
                'date': window_date,
                'n_factors': len(selection),
                'factors': selection['factor'].tolist() if len(selection) > 0 else [],
                'grs_stat': selection.iloc[-1]['grs_statistic'] if len(selection) > 0 else np.nan,
                'avg_alpha': selection.iloc[-1]['avg_abs_alpha'] if len(selection) > 0 else np.nan
            }
            results.append(window_result)
            
            print(f"  {window_date.strftime('%Y-%m')}: {len(selection)} facteurs sélectionnés")
        
        self.adaptive_selection = pd.DataFrame(results)
        return self.adaptive_selection
    
    def analyze_temporal_stability(self) -> pd.DataFrame:
        """
        Analyse la stabilité temporelle des facteurs sélectionnés.
        """
        if self.adaptive_selection is None:
            self.adaptive_factor_selection()
        
        # Fréquence de sélection de chaque facteur
        all_selections = []
        for factors in self.adaptive_selection['factors']:
            all_selections.extend(factors)
        
        factor_frequency = pd.Series(all_selections).value_counts() / len(self.adaptive_selection)
        
        # Persistance temporelle
        factor_persistence = {}
        for factor in self.factor_names:
            consecutive_counts = []
            current_streak = 0
            
            for factors in self.adaptive_selection['factors']:
                if factor in factors:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        consecutive_counts.append(current_streak)
                    current_streak = 0
            
            if current_streak > 0:
                consecutive_counts.append(current_streak)
            
            factor_persistence[factor] = {
                'avg_streak': np.mean(consecutive_counts) if consecutive_counts else 0,
                'max_streak': np.max(consecutive_counts) if consecutive_counts else 0,
                'n_appearances': len(consecutive_counts)
            }
        
        # Créer le DataFrame de résultats
        stability_df = pd.DataFrame(factor_persistence).T
        stability_df['selection_frequency'] = factor_frequency
        stability_df = stability_df.fillna(0)
        
        self.temporal_stability = stability_df
        return stability_df
    
    def simple_regime_analysis(self) -> Dict:
        """
        Analyse simple des régimes basée sur la volatilité du marché.
        """
        # Calculer la volatilité rolling
        market_vol = self.market_return.rolling(window=21).std()
        
        # Définir les régimes basés sur la volatilité
        vol_threshold = market_vol.median()
        high_vol_regime = market_vol > vol_threshold
        
        # Analyser les facteurs par régime
        regime_analysis = {}
        
        for regime, mask in [('Low_Vol', ~high_vol_regime), ('High_Vol', high_vol_regime)]:
            regime_data = []
            
            for factor in self.factor_names:
                # Performance du facteur dans ce régime
                factor_returns = self.factors_df[factor][mask]
                
                regime_data.append({
                    'factor': factor,
                    'mean_return': factor_returns.mean() * 12 * 100,  # Annualisé en %
                    'volatility': factor_returns.std() * np.sqrt(12) * 100,
                    'sharpe': factor_returns.mean() / factor_returns.std() * np.sqrt(12)
                })
            
            regime_analysis[regime] = pd.DataFrame(regime_data)
        
        return regime_analysis
    
    def plot_adaptive_selection(self) -> plt.Figure:
        """Visualise la sélection adaptative de facteurs."""
        if self.adaptive_selection is None:
            self.adaptive_factor_selection()
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Nombre de facteurs sélectionnés
        ax1 = axes[0]
        ax1.plot(self.adaptive_selection['date'], 
                self.adaptive_selection['n_factors'], 
                marker='o', linewidth=2)
        ax1.set_ylabel('Nombre de facteurs')
        ax1.set_title('Évolution du nombre de facteurs sélectionnés')
        ax1.grid(True, alpha=0.3)
        
        # Alpha moyen
        ax2 = axes[1]
        ax2.plot(self.adaptive_selection['date'], 
                self.adaptive_selection['avg_alpha'], 
                marker='o', linewidth=2, color='green')
        ax2.set_ylabel('Alpha moyen (%)')
        ax2.set_title('Évolution de l\'alpha moyen des facteurs restants')
        ax2.grid(True, alpha=0.3)
        
        # Heatmap des facteurs sélectionnés
        ax3 = axes[2]
        
        # Créer la matrice de sélection
        unique_factors = list(set(sum(self.adaptive_selection['factors'].tolist(), [])))
        selection_matrix = np.zeros((len(unique_factors), len(self.adaptive_selection)))
        
        for i, date_factors in enumerate(self.adaptive_selection['factors']):
            for factor in date_factors:
                if factor in unique_factors:
                    j = unique_factors.index(factor)
                    selection_matrix[j, i] = 1
        
        # Sélectionner les facteurs les plus fréquents
        factor_freq = selection_matrix.sum(axis=1)
        top_factors_idx = np.argsort(factor_freq)[-20:]  # Top 20
        
        im = ax3.imshow(selection_matrix[top_factors_idx], 
                       aspect='auto', cmap='Blues')
        ax3.set_yticks(range(len(top_factors_idx)))
        ax3.set_yticklabels([unique_factors[i] for i in top_factors_idx])
        ax3.set_xlabel('Fenêtre temporelle')
        ax3.set_title('Facteurs sélectionnés dans le temps (Top 20)')
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_stability(self) -> plt.Figure:
        """Visualise la stabilité temporelle des facteurs."""
        if self.temporal_stability is None:
            self.analyze_temporal_stability()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top facteurs par fréquence
        ax1 = axes[0, 0]
        top_freq = self.temporal_stability.nlargest(15, 'selection_frequency')
        bars = ax1.barh(range(len(top_freq)), top_freq['selection_frequency'])
        ax1.set_yticks(range(len(top_freq)))
        ax1.set_yticklabels(top_freq.index)
        ax1.set_xlabel('Fréquence de sélection')
        ax1.set_title('Top 15 facteurs par fréquence')
        ax1.grid(True, alpha=0.3)
        
        # Couleur par cluster si disponible
        try:
            from clusters import create_factor_clusters
            get_cluster = create_factor_clusters()
            colors = []
            for factor in top_freq.index:
                cluster = get_cluster(factor)
                if cluster == 'Quality':
                    colors.append('red')
                elif cluster == 'Value':
                    colors.append('blue')
                elif cluster == 'Momentum':
                    colors.append('green')
                elif cluster == 'Low Risk':
                    colors.append('orange')
                elif cluster == 'Investment':
                    colors.append('purple')
                else:
                    colors.append('gray')
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        except:
            pass
        
        # Persistance moyenne
        ax2 = axes[0, 1]
        top_persist = self.temporal_stability.nlargest(15, 'avg_streak')
        ax2.barh(range(len(top_persist)), top_persist['avg_streak'])
        ax2.set_yticks(range(len(top_persist)))
        ax2.set_yticklabels(top_persist.index)
        ax2.set_xlabel('Persistance moyenne (fenêtres)')
        ax2.set_title('Top 15 facteurs par persistance')
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot fréquence vs persistance
        ax3 = axes[1, 0]
        x = self.temporal_stability['selection_frequency']
        y = self.temporal_stability['avg_streak']
        ax3.scatter(x, y, alpha=0.6)
        
        # Annoter les points importants
        top_factors = self.temporal_stability.nlargest(10, 'selection_frequency').index
        for factor in top_factors:
            ax3.annotate(factor, 
                        (self.temporal_stability.loc[factor, 'selection_frequency'],
                         self.temporal_stability.loc[factor, 'avg_streak']),
                        fontsize=8)
        
        ax3.set_xlabel('Fréquence de sélection')
        ax3.set_ylabel('Persistance moyenne')
        ax3.set_title('Fréquence vs Persistance')
        ax3.grid(True, alpha=0.3)
        
        # Distribution de la fréquence de sélection
        ax4 = axes[1, 1]
        freq_data = self.temporal_stability['selection_frequency']
        freq_data = freq_data[freq_data > 0]  # Exclure les facteurs jamais sélectionnés
        ax4.hist(freq_data, bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(freq_data.mean(), color='red', linestyle='--', 
                   label=f'Moyenne: {freq_data.mean():.2f}')
        ax4.set_xlabel('Fréquence de sélection')
        ax4.set_ylabel('Nombre de facteurs')
        ax4.set_title('Distribution des fréquences de sélection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def run_temporal_analysis(factors_df: pd.DataFrame, market_return: pd.Series) -> Dict:
    """
    Exécute l'analyse temporelle sur les données des facteurs.
    
    Args:
        factors_df: DataFrame contenant les facteurs
        market_return: Série des rendements du marché
        
    Returns:
        Dict: Résultats de l'analyse
    """
    print("\n=== Analyse Temporelle Avancée du Factor Zoo ===\n")
    
    # Initialiser l'objet d'analyse
    temporal_analysis = TemporalDynamicsAnalysis(factors_df, market_return)
    
    # 1. Sélection adaptative
    adaptive_selection = temporal_analysis.adaptive_factor_selection(
        window_size=60,  # 5 ans
        step_size=12     # 1 an
    )
    
    # 2. Analyse de stabilité temporelle
    temporal_stability = temporal_analysis.analyze_temporal_stability()
    
    # 3. Analyse simple des régimes
    regime_analysis = temporal_analysis.simple_regime_analysis()
    
    # 4. Créer les visualisations
    adaptive_plot = temporal_analysis.plot_adaptive_selection()
    stability_plot = temporal_analysis.plot_temporal_stability()
    
    # Sauvegarder les figures
    adaptive_plot.savefig('temporal_adaptive_selection.png', dpi=300, bbox_inches='tight')
    stability_plot.savefig('temporal_stability.png', dpi=300, bbox_inches='tight')
    
    print("\n=== Analyse Temporelle terminée ===")
    
    # Afficher quelques statistiques clés
    print("\nRésumé des résultats:")
    print(f"- Nombre moyen de facteurs sélectionnés: {adaptive_selection['n_factors'].mean():.1f}")
    print(f"- Écart-type du nombre de facteurs: {adaptive_selection['n_factors'].std():.1f}")
    
    top_5_stable = temporal_stability.nlargest(5, 'selection_frequency')
    print("\nTop 5 facteurs les plus stables:")
    for factor, row in top_5_stable.iterrows():
        print(f"  {factor}: fréquence={row['selection_frequency']:.2%}, persistance={row['avg_streak']:.1f}")
    
    # Retourner les résultats
    return {
        'adaptive_selection': adaptive_selection,
        'temporal_stability': temporal_stability,
        'regime_analysis': regime_analysis,
        'temporal_object': temporal_analysis
    }
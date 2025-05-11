# exhibits_complete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from factor_selection import IterativeFactorSelection
from data_loader import DataLoader
from clusters import create_factor_clusters
from visualisation import FactorZooPlotter


class FactorZooExhibits:
    """Reproduction complète de TOUS les exhibits de l'article Factor Zoo"""
    
    def __init__(self, start_date="1971-11-01", end_date="2021-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        self.results_dict = {}
        self.weighting_schemes = ['VW_cap', 'VW', 'EW']  # TOUS les schémas
        self.regions = ['US', 'World', 'World_ex_US']  # TOUTES les régions
        self.get_cluster = create_factor_clusters()
        
    def prepare_all_data(self):
        """Prépare les données pour TOUS les schémas et régions"""
        print("=== Préparation des données pour tous les schémas et régions ===")
        
        # 1. US avec différents schémas de pondération
        for scheme in self.weighting_schemes:
            print(f"\nTraitement US - Schéma : {scheme}")
            try:
                selector = IterativeFactorSelection(
                    scheme, 
                    self.start_date, 
                    self.end_date,
                    region_factors_X='world', 
                    region_factors_y='US'
                )
                results = selector.select_factors_t_std(max_factors=30)
                self.results_dict[f'{scheme}_US'] = results
            except Exception as e:
                print(f"Erreur pour {scheme}_US: {e}")
        
        # 2. Différentes régions avec VW_cap (Exhibit 8)
        # Utiliser les noms corrects pour les régions
        region_mapping = {
            'US': 'US',
            'World': 'world',
            'World_ex_US': 'ex US'
        }
        
        for region_key, region_value in region_mapping.items():
            print(f"\nTraitement {region_key} - Schéma : VW_cap")
            try:
                selector = IterativeFactorSelection(
                    'VW_cap',
                    '1993-08-01',  # Période plus courte pour l'international
                    self.end_date,
                    region_factors_X='world',
                    region_factors_y=region_value
                )
                results = selector.select_factors_t_std(max_factors=30)
                self.results_dict[f'VW_cap_{region_key}'] = results
            except Exception as e:
                print(f"Erreur pour VW_cap_{region_key}: {e}")
    
    def exhibit_1_factor_alphas(self):
        """Exhibit 1: Factor Alphas - Graphique des alphas CAPM"""
        print("\nGénération Exhibit 1: Factor Alphas")
        
        data_loader = DataLoader('VW_cap', self.start_date, self.end_date)
        factors_df, market_return = data_loader.load_factor_data('US')
        
        # Calculer les alphas CAPM pour tous les facteurs
        alphas = {}
        for factor in factors_df.columns:
            y = factors_df[factor]
            X = sm.add_constant(market_return)
            valid_idx = ~(y.isna() | market_return.isna())
            
            if valid_idx.sum() > 30:
                model = sm.OLS(y[valid_idx], X[valid_idx]).fit()
                alphas[factor] = model.params[0] * 12 * 100  # Annualisé en %
        
        # Préparer les données pour le graphique
        alpha_df = pd.DataFrame.from_dict(alphas, orient='index', columns=['alpha'])
        alpha_df['cluster'] = alpha_df.index.map(self.get_cluster)
        alpha_df = alpha_df.sort_values(['cluster', 'alpha'])
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Couleurs par cluster
        colors = plt.cm.tab20(np.linspace(0, 1, len(alpha_df['cluster'].unique())))
        cluster_colors = dict(zip(alpha_df['cluster'].unique(), colors))
        
        # Barres par cluster
        x_pos = 0
        x_positions = []
        x_labels = []
        
        for cluster in alpha_df['cluster'].unique():
            cluster_data = alpha_df[alpha_df['cluster'] == cluster]
            
            for idx, row in cluster_data.iterrows():
                color = cluster_colors.get(cluster, '#333333')
                ax.bar(x_pos, row['alpha'], width=0.8, color=color, alpha=0.8)
                x_positions.append(x_pos)
                x_labels.append(idx)
                x_pos += 1
            
            x_pos += 0.5  # Espace entre clusters
        
        ax.set_ylabel('Alphas p.a. [%]', fontsize=12)
        ax.set_title('Factor Alphas', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Labels sur l'axe x
        step = max(1, len(x_positions) // 30)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels(x_labels[::step], rotation=90, fontsize=8)
        
        # Légende
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8, label=cluster) 
                          for cluster, color in cluster_colors.items()]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=7, frameon=False)
        
        plt.tight_layout()
        return fig
    
    def exhibit_2_selection_table(self):
        """Exhibit 2: Iterative Factor Selection Table (VW_cap US)"""
        print("\nGénération Exhibit 2: Selection Table")
        
        results = self.results_dict.get('VW_cap_US')
        if results is None:
            return None
        
        exhibit_2 = pd.DataFrame()
        exhibit_2['No.'] = results['iteration']
        exhibit_2['Factor'] = results['factor']
        exhibit_2['Cluster'] = results['factor'].apply(self.get_cluster)
        exhibit_2['GRS'] = results['grs_statistic'].map('{:.2f}'.format)
        exhibit_2['p(GRS)'] = results['grs_pvalue'].map('{:.3f}'.format)
        exhibit_2['Avg|α|'] = results['avg_abs_alpha'].map('{:.2f}'.format)
        exhibit_2['Sh²(f)'] = results['sh2_f'].map('{:.2f}'.format)
        exhibit_2['SR'] = results['sr'].map('{:.2f}'.format)
        exhibit_2['n(α)ₜ>₂'] = results['n_significant_t2']
        exhibit_2['n(α)ₜ>₃'] = results['n_significant_t3']
        
        return exhibit_2
    
    def exhibit_3_scheme_comparison_table(self):
        """Exhibit 3: Comparison of Different Weighting Schemes"""
        print("\nGénération Exhibit 3: Scheme Comparison Table")
        
        # Créer un tableau similaire à l'Exhibit 2 de l'article
        summary_data = []
        
        for scheme in self.weighting_schemes:
            results = self.results_dict.get(f'{scheme}_US')
            if results is None:
                continue
            
            # Premier facteur où n(α)ₜ>₃ = 0
            t3_zero = results[results['n_significant_t3'] == 0]
            t3_iter = t3_zero.iloc[0]['iteration'] if not t3_zero.empty else '>30'
            
            # Premier facteur où n(α)ₜ>₂ = 0
            t2_zero = results[results['n_significant_t2'] == 0]
            t2_iter = t2_zero.iloc[0]['iteration'] if not t2_zero.empty else '>30'
            
            # Premier facteur où p(GRS) > 0.05
            grs_pass = results[results['grs_pvalue'] > 0.05]
            grs_iter = grs_pass.iloc[0]['iteration'] if not grs_pass.empty else '>30'
            
            summary_data.append({
                'Weighting Scheme': scheme,
                'Total Factors': len(results),
                'No. Factors (t > 3)': t3_iter,
                'No. Factors (t > 2)': t2_iter,
                'No. Factors (GRS 5%)': grs_iter,
                'Final GRS': f"{results.iloc[-1]['grs_statistic']:.2f}",
                'Final p(GRS)': f"{results.iloc[-1]['grs_pvalue']:.3f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def exhibit_4_selected_factors_plot(self):
        """Exhibit 4: Selected Alpha Factors Plot"""
        print("\nGénération Exhibit 4: Selected Factors Plot")
        
        # Utiliser FactorZooPlotter pour créer le graphique
        results = self.results_dict.get('VW_cap_US')
        if results is None:
            return None
        
        # Charger les données
        data_loader = DataLoader('VW_cap', self.start_date, self.end_date)
        factors_df, market_return = data_loader.load_factor_data('US')
        
        # Facteurs sélectionnés (top 15)
        selected_factors = results.head(15)['factor'].tolist()
        
        # Clusters pour tous les facteurs
        clusters = {factor: self.get_cluster(factor) for factor in factors_df.columns}
        
        # Utiliser la méthode de visualisation
        plotter = FactorZooPlotter()
        fig = plotter.plot_alphas_chart(factors_df, selected_factors, clusters, market_return)
        
        return fig
    
    def exhibit_5_factor_persistence(self):
        """Exhibit 5: Factor Persistence (simplifié sans extension temporelle)"""
        print("\nGénération Exhibit 5: Factor Persistence (version simplifiée)")
        
        # Version simplifiée montrant la fréquence de sélection des facteurs
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compter combien de fois chaque facteur apparaît dans différents schémas
        factor_counts = {}
        for key, results in self.results_dict.items():
            if '_US' in key:  # Seulement les résultats US
                for _, row in results.iterrows():
                    factor = row['factor']
                    factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Trier par fréquence
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        factors = [f[0] for f in sorted_factors]
        counts = [f[1] for f in sorted_factors]
        
        bars = ax.bar(range(len(factors)), counts, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels(factors, rotation=45, ha='right')
        ax.set_ylabel('Frequency of Selection')
        ax.set_title('Factor Selection Frequency Across Weighting Schemes')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def exhibit_6_grs_evolution(self):
        """Exhibit 6: GRS Evolution Across Schemes"""
        print("\nGénération Exhibit 6: GRS Evolution")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Factor Selection Metrics by Weighting Scheme', fontsize=16)
        
        colors = {'VW_cap': 'blue', 'VW': 'red', 'EW': 'green'}
        
        for scheme in self.weighting_schemes:
            results = self.results_dict.get(f'{scheme}_US')
            if results is None:
                continue
            
            color = colors.get(scheme, 'black')
            
            # GRS Statistic
            ax1.plot(results['iteration'], results['grs_statistic'], 
                    'o-', linewidth=2, label=scheme, color=color)
            ax1.set_ylabel('GRS Statistic')
            ax1.set_title('GRS Statistic')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # p-value
            ax2.plot(results['iteration'], results['grs_pvalue'], 
                    'o-', linewidth=2, label=scheme, color=color)
            ax2.set_ylabel('p-value')
            ax2.set_title('GRS p-value')
            ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # n(α)ₜ>₃
            ax3.plot(results['iteration'], results['n_significant_t3'], 
                    'o-', linewidth=2, label=scheme, color=color)
            ax3.set_xlabel('Number of Factors')
            ax3.set_ylabel('n(α)ₜ>₃')
            ax3.set_title('Significant Factors (|t| > 3)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Sharpe²
            ax4.plot(results['iteration'], results['sh2_f'], 
                    'o-', linewidth=2, label=scheme, color=color)
            ax4.set_xlabel('Number of Factors')
            ax4.set_ylabel('Sh²(f)')
            ax4.set_title('Squared Sharpe Ratio')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def exhibit_7_scheme_comparison_plot(self):
        """Exhibit 7: Direct Comparison Plot (correspond à l'Exhibit 7 de l'article)"""
        print("\nGénération Exhibit 7: Scheme Comparison Plot")
        
        # Utiliser FactorZooPlotter
        plotter = FactorZooPlotter()
        
        # Préparer les résultats pour les 3 schémas
        results_dict = {}
        for scheme in self.weighting_schemes:
            results = self.results_dict.get(f'{scheme}_US')
            if results is not None:
                results_dict[scheme] = results
        
        # Créer le graphique de comparaison
        fig = plotter.plot_comparison(results_dict)
        return fig
    
    def exhibit_8_international_comparison(self):
        """Exhibit 8: International Comparison"""
        print("\nGénération Exhibit 8: International Comparison")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('International Factor Selection Comparison', fontsize=16)
        
        colors = {'US': 'blue', 'World': 'green', 'World_ex_US': 'red'}
        labels = {'US': 'US', 'World': 'World', 'World_ex_US': 'World ex US'}
        
        for region in self.regions:
            results = self.results_dict.get(f'VW_cap_{region}')
            if results is None:
                continue
            
            color = colors.get(region, 'black')
            label = labels.get(region, region)
            
            # GRS Statistic
            ax1.plot(results['iteration'], results['grs_statistic'], 
                    'o-', linewidth=2, label=label, color=color)
            ax1.set_ylabel('GRS Statistic')
            ax1.set_title('GRS Statistic by Region')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # p-value
            ax2.plot(results['iteration'], results['grs_pvalue'], 
                    'o-', linewidth=2, label=label, color=color)
            ax2.set_ylabel('p-value')
            ax2.set_title('GRS p-value by Region')
            ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Average |α|
            ax3.plot(results['iteration'], results['avg_abs_alpha'], 
                    'o-', linewidth=2, label=label, color=color)
            ax3.set_xlabel('Number of Factors')
            ax3.set_ylabel('Avg|α| (%)')
            ax3.set_title('Average Absolute Alpha by Region')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Sharpe²
            ax4.plot(results['iteration'], results['sh2_f'], 
                    'o-', linewidth=2, label=label, color=color)
            ax4.set_xlabel('Number of Factors')
            ax4.set_ylabel('Sh²(f)')
            ax4.set_title('Squared Sharpe Ratio by Region')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def exhibit_9_regional_summary_table(self):
        """Exhibit 9: Regional Summary Table"""
        print("\nGénération Exhibit 9: Regional Summary")
        
        summary_data = []
        
        for region in self.regions:
            results = self.results_dict.get(f'VW_cap_{region}')
            if results is None:
                continue
            
            # Statistiques similaires à l'Exhibit 3
            t3_zero = results[results['n_significant_t3'] == 0]
            t3_iter = t3_zero.iloc[0]['iteration'] if not t3_zero.empty else '>30'
            
            t2_zero = results[results['n_significant_t2'] == 0]
            t2_iter = t2_zero.iloc[0]['iteration'] if not t2_zero.empty else '>30'
            
            grs_pass = results[results['grs_pvalue'] > 0.05]
            grs_iter = grs_pass.iloc[0]['iteration'] if not grs_pass.empty else '>30'
            
            summary_data.append({
                'Region': region.replace('_', ' '),
                'Total Factors': len(results),
                'No. Factors (t > 3)': t3_iter,
                'No. Factors (t > 2)': t2_iter,
                'No. Factors (GRS 5%)': grs_iter,
                'Initial GRS': f"{results.iloc[0]['grs_statistic']:.2f}",
                'Final GRS': f"{results.iloc[-1]['grs_statistic']:.2f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def save_all_exhibits(self, output_dir='exhibits_complete'):
        """Sauvegarde TOUS les exhibits"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Préparer toutes les données
        self.prepare_all_data()
        
        # Exhibit 1: Factor Alphas
        print("\n=== Sauvegarde Exhibit 1 ===")
        fig1 = self.exhibit_1_factor_alphas()
        fig1.savefig(f'{output_dir}/exhibit_1_factor_alphas.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Exhibit 2: Selection Table
        print("\n=== Sauvegarde Exhibit 2 ===")
        table2 = self.exhibit_2_selection_table()
        table2.to_csv(f'{output_dir}/exhibit_2_selection_table.csv', index=False)
        print(table2.head(15))
        
        # Exhibit 3: Scheme Comparison Table
        print("\n=== Sauvegarde Exhibit 3 ===")
        table3 = self.exhibit_3_scheme_comparison_table()
        table3.to_csv(f'{output_dir}/exhibit_3_scheme_comparison.csv', index=False)
        print(table3)
        
        # Exhibit 4: Selected Factors Plot
        print("\n=== Sauvegarde Exhibit 4 ===")
        fig4 = self.exhibit_4_selected_factors_plot()
        if fig4:
            fig4.savefig(f'{output_dir}/exhibit_4_selected_factors.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
        
        # Exhibit 5: Factor Persistence
        print("\n=== Sauvegarde Exhibit 5 ===")
        fig5 = self.exhibit_5_factor_persistence()
        fig5.savefig(f'{output_dir}/exhibit_5_factor_persistence.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        # Exhibit 6: GRS Evolution
        print("\n=== Sauvegarde Exhibit 6 ===")
        fig6 = self.exhibit_6_grs_evolution()
        fig6.savefig(f'{output_dir}/exhibit_6_grs_evolution.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
        
        # Exhibit 7: Scheme Comparison Plot
        print("\n=== Sauvegarde Exhibit 7 ===")
        fig7 = self.exhibit_7_scheme_comparison_plot()
        if fig7:
            fig7.savefig(f'{output_dir}/exhibit_7_scheme_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig7)
        
        # Exhibit 8: International Comparison
        print("\n=== Sauvegarde Exhibit 8 ===")
        fig8 = self.exhibit_8_international_comparison()
        fig8.savefig(f'{output_dir}/exhibit_8_international_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig8)
        
        # Exhibit 9: Regional Summary
        print("\n=== Sauvegarde Exhibit 9 ===")
        table9 = self.exhibit_9_regional_summary_table()
        table9.to_csv(f'{output_dir}/exhibit_9_regional_summary.csv', index=False)
        print(table9)
        
        print(f"\n=== TOUS les exhibits ont été sauvegardés dans '{output_dir}' ===")
        
        # Résumé final
        print("\n=== RÉSUMÉ DES RÉSULTATS ===")
        print("\n1. Comparaison des schémas de pondération (US):")
        for scheme in self.weighting_schemes:
            results = self.results_dict.get(f'{scheme}_US')
            if results:
                n_factors = len(results[results['n_significant_t3'] == 0].head(1))
                if n_factors > 0:
                    final_iter = results[results['n_significant_t3'] == 0].iloc[0]['iteration']
                    print(f"   {scheme}: {final_iter} facteurs nécessaires")
        
        print("\n2. Comparaison internationale (VW_cap):")
        for region in self.regions:
            results = self.results_dict.get(f'VW_cap_{region}')
            if results:
                n_factors = len(results[results['n_significant_t3'] == 0].head(1))
                if n_factors > 0:
                    final_iter = results[results['n_significant_t3'] == 0].iloc[0]['iteration']
                    print(f"   {region}: {final_iter} facteurs nécessaires")


# Utilisation
if __name__ == "__main__":
    print("=== Reproduction complète des exhibits du Factor Zoo ===")
    print("Ceci peut prendre plusieurs minutes...")
    
    exhibits = FactorZooExhibits(start_date="1971-11-01", end_date="2021-12-31")
    exhibits.save_all_exhibits()
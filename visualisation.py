import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from clusters import create_factor_clusters

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
    
    def create_exhibit_comparison(your_results, formatted=True):
        """Crée un tableau comparable à l'Exhibit 2 de l'article."""
        
        # Structure similaire à l'article
        exhibit = pd.DataFrame()
        
        exhibit['No.'] = your_results['iteration']
        exhibit['Factor'] = your_results['factor']
        
        # Clusters (vous devez les mapper)
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
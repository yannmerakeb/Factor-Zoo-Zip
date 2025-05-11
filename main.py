import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from factor_selection import IterativeFactorSelection
from clusters import create_factor_clusters
from visualisation import FactorZooPlotter

def main(weighting_schemes : list):

    results_dict = {}

    # Créer la fonction de mapping pour les clusters
    get_cluster = create_factor_clusters()
    
    # Dictionnaire pour stocker les facteurs sélectionnés par schéma
    selected_factors_dict = {}
    factors_df_dict = {}
    market_return_dict = {}
    
    for scheme in weighting_schemes:
        print(f"\nTraitement du schéma de pondération: {scheme}")

        # Charger les données
        data_loader = DataLoader(scheme, '1993-08-01', '2021-12-31')
        factors_df, market_return = data_loader.load_factor_data('US')

        # Stocker pour utilisation ultérieure
        factors_df_dict[scheme] = factors_df
        market_return_dict[scheme] = market_return
        
        # Sélection itérative
        selector = IterativeFactorSelection(factors_df, market_return)
        
        results = selector.select_factors_t_std()
        results_dict[scheme] = results
        
        # Stocker les facteurs sélectionnés
        selected_factors_dict[scheme] = results['factor'].tolist()
    
    # Créer la table des facteurs (similaire à l'Exhibit 3)
    factor_table = FactorZooPlotter.create_exhibit_table(results_dict)
    print("\nTableau des facteurs sélectionnés:")
    print(factor_table)
    
    # Créer le tableau récapitulatif
    summary_df = FactorZooPlotter.create_summary_table(results_dict)
    print("\nTableau récapitulatif:")
    print(summary_df)
    
    # Comparer les schémas (similaire aux comparaisons dans l'article)
    fig = FactorZooPlotter.plot_comparison(results_dict)
    plt.show()

    # Créer le graphique des alphas (similaire à l'Exhibit 4) pour chaque schéma
    """for scheme in weighting_schemes:
        # Attribuer les clusters à tous les facteurs
        factor_clusters = {factor: get_cluster(factor) for factor in factors_df_dict[scheme].columns}
        
        # Créer le graphique des alphas
        fig = FactorZooPlotter.plot_alphas_chart(
            factors_df_dict[scheme],
            selected_factors_dict[scheme],
            factor_clusters,
            market_return_dict[scheme]
        )
        plt.show()"""
    
    return results_dict, summary_df, factor_table

if __name__ == "__main__":
    
    # Exécuter l'analyse pour tous les schémas
    results_dict, summary_df, factor_table = main(weighting_schemes=['VW_cap', 'EW', 'VW'])
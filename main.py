import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from factor_zoo import FactorZooPlotter, DataLoader, IterativeFactorSelection
from clusters import create_factor_clusters

def main(data_paths : dict, weighting_schemes : list):

    results_dict = {}

    # Créer la fonction de mapping pour les clusters
    get_cluster = create_factor_clusters()
    
    # Dictionnaire pour stocker les facteurs sélectionnés par schéma
    selected_factors_dict = {}
    factors_df_dict = {}
    market_return_dict = {}
    
    for scheme in weighting_schemes:
        print(f"\nTraitement du schéma de pondération: {scheme}")
        data_path = data_paths[scheme]
        print(f"Utilisation du fichier: {data_path}")
        
        # Charger les données
        data_loader = DataLoader(data_path)
        factors_df, market_return = data_loader.load_factor_data()
        
        # Stocker pour utilisation ultérieure
        factors_df_dict[scheme] = factors_df
        market_return_dict[scheme] = market_return
        
        # Sélection itérative
        selector = IterativeFactorSelection(factors_df, market_return)
        results = selector.select_factors()
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
    for scheme in weighting_schemes:
        # Attribuer les clusters à tous les facteurs
        factor_clusters = {factor: get_cluster(factor) for factor in factors_df_dict[scheme].columns}
        
        # Créer le graphique des alphas
        fig = FactorZooPlotter.plot_alphas_chart(
            factors_df_dict[scheme],
            selected_factors_dict[scheme],
            factor_clusters,
            market_return_dict[scheme]
        )
        plt.show()
    
    return results_dict, summary_df, factor_table

if __name__ == "__main__":
    
    data_paths = {
        'VW Cap': "data/[usa]_[all_factors]_[monthly]_[vw_cap].csv",
        'EW': "data/[usa]_[all_factors]_[monthly]_[ew].csv",
        'VW': "data/[usa]_[all_factors]_[monthly]_[vw].csv"
    }
    
    # Exécuter l'analyse pour tous les schémas
    results_dict, summary_df, factor_table = main(data_paths, ['VW Cap', 'EW', 'VW'])
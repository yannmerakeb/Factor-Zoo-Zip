import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from factor_selection import IterativeFactorSelection
from visualisation import FactorZooPlotter
from main import main

def extension_latest_data(schemes : list):
    """Fonction pour exécuter l'analyse sur les données les plus récentes."""

    results_dict, summary_df, factor_table = main(weighting_schemes=schemes, 
                                                  start_date='1993-08-01', 
                                                  end_date='2024-12-31',
                                                  region= 'world')

    return results_dict, summary_df, factor_table

def aggregate_crisis_periods(df, crisis_periods):
    """Fonction pour identifier les périodes de crise dans un DataFrame temporel."""
    
    df['is_crisis'] = False
    
    for crisis_name, (start_date, end_date) in crisis_periods.items():
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Marquer les périodes de crise
        crisis_mask = (df.index >= start_date) & (df.index <= end_date)
        df.loc[crisis_mask, 'is_crisis'] = True
        
    return df

def extension_all_crisis_data(schemes: list, crisis_periods: dict):
    """Fonction pour exécuter l'analyse uniquement sur l'ensemble des périodes de crise agrégées."""
    
    # Nous avons besoin d'extraire les données pour identifier les périodes de crise
    sample_scheme = schemes[0]
    start_date_full = '1993-08-01'
    end_date_full = '2024-12-31'
    
    try:
        # Accéder à toutes les données
        data_loader = DataLoader(sample_scheme, start_date_full, end_date_full)
        factors_df, market_return = data_loader.load_factor_data("ex US")

        # Combiner les séries temporelles dans un DataFrame
        full_data = pd.DataFrame(index=factors_df.index)
        full_data['market_return'] = market_return
        
        # Identifier les périodes de crise
        full_data = aggregate_crisis_periods(full_data, crisis_periods)
        
        # Créer des indices distincts pour les périodes de crise
        crisis_dates = full_data[full_data['is_crisis']].index.tolist()
        print(f"Périodes de crise identifiées: {len(crisis_dates)} mois")
            
        # Maintenant, exécuter l'analyse principale sur ces dates spécifiques
        results_dict = {}
        selected_factors = {}
        
        for scheme in schemes:
            print(f"\nTraitement du schéma de pondération pour les crises: {scheme}")
            
            # Réinitialiser le sélecteur pour chaque schéma
            crisis_selector = IterativeFactorSelection(
                factors_df.loc[crisis_dates],
                market_return.loc[crisis_dates],
            )
            
            # Vérifier si nous avons suffisamment de données
            if len(factors_df.loc[crisis_dates]) < 36:  # Minimum de 36 mois pour l'analyse
                print(f"Attention: Seulement {len(crisis_selector.factors_df)} mois de données de crise.")
                print("L'analyse pourrait ne pas être fiable en raison du petit échantillon.")
            
            # Exécuter la sélection de facteurs
            try:
                results = crisis_selector.select_factors_t_std()
                results_dict[scheme] = results
                selected_factors[scheme] = results['factor'].tolist()
            except Exception as e:
                print(f"Erreur lors de la sélection des facteurs pour {scheme}: {e}")
                results_dict[scheme] = pd.DataFrame()
        
        # Créer les tableaux récapitulatifs
        if any(not df.empty for df in results_dict.values()):
            factor_table = FactorZooPlotter.create_exhibit_table(results_dict)
            summary_df = FactorZooPlotter.create_summary_table(results_dict)
            
            # Ajouter une colonne pour identifier que c'est la période de crise agrégée
            summary_df['Period'] = 'All_Crisis'
            
            print("\nTableau des facteurs sélectionnés (périodes de crise):")
            print(factor_table)
            
            print("\nTableau récapitulatif (périodes de crise):")
            print(summary_df)
            
            return results_dict, summary_df, factor_table
        else:
            print("Aucun résultat valide n'a été obtenu pour les périodes de crise.")
            return None, None, None
            
    except Exception as e:
        print(f"Erreur lors de l'analyse des périodes de crise: {e}")
        return None, None, None

if __name__ == "__main__":
    schemes = ['VW_cap', 'EW', 'VW']
    
    # Définir les périodes de crise
    crisis_periods = {
        'Subprimes': ('2007-01-31', '2010-12-31'),
        'Dotcom': ('2000-03-31', '2002-10-31'),
        'COVID': ('2020-01-31', '2021-06-30'),
        'Crise_2022': ('2021-12-31', '2023-06-30')
    }
    
    # Analyse sur les périodes de crise agrégées
    print("\n" + "="*50)
    print("ANALYSE SUR L'AGRÉGATION DES PÉRIODES DE CRISE")
    print("="*50)
    crisis_results, crisis_summary, crisis_table = extension_all_crisis_data(schemes, crisis_periods)


    print("ANALYSE SUR LES DONNÉES LES PLUS RÉCENTES")
    print("="*50)
    # Analyse sur les données les plus récentes
    latest_results, latest_summary, latest_table = extension_latest_data(schemes)
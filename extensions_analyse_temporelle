import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from factor_selection import IterativeFactorSelection
from visualisation import FactorZooPlotter
from main import main

def extension_base_data(schemes: list):
    """Fonction pour exécuter l'analyse sur les données de base."""
    
    start_date = '1993-08-01'
    end_date = '2021-12-31'
    
    print(f"Analyse de la période de base: {start_date} à {end_date}")
    results_dict, summary_df, factor_table = main(
        weighting_schemes=schemes, 
        start_date=start_date,
        end_date=end_date
    )
    
    return results_dict, summary_df, factor_table

def extension_latest_data(schemes : list):
    """Fonction pour exécuter l'analyse sur les données les plus récentes."""

    results_dict, summary_df, factor_table = main(weighting_schemes=schemes, start_date='1993-08-01', end_date='2024-12-31')

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
    # On utilise temporairement le premier schéma pour obtenir les données
    sample_scheme = schemes[0]
    start_date_full = '1993-08-01'  # Même période que la base
    end_date_full = '2021-12-31'
    
    try:
        # Créer une instance temporaire du sélecteur pour accéder aux données
        temp_selector = IterativeFactorSelection(
            sample_scheme,
            start_date_full,
            end_date_full
        )
        
        # Accéder aux données de facteurs et au rendement du marché
        factors_df = temp_selector.factors_df.copy()
        market_return = temp_selector.market_return.copy()
        
        # Combiner les séries temporelles dans un DataFrame
        full_data = pd.DataFrame(index=factors_df.index)
        full_data['market_return'] = market_return
        
        # Identifier les périodes de crise
        full_data = aggregate_crisis_periods(full_data, crisis_periods)
        
        # Créer des indices distincts pour les périodes de crise
        crisis_dates = full_data[full_data['is_crisis']].index.tolist()
        
        if not crisis_dates:
            print("Aucune période de crise n'a été identifiée dans les données.")
            return None, None, None
        
        print(f"Périodes de crise identifiées: {len(crisis_dates)} mois")
        
        # Convertir les dates en chaînes de caractères pour l'enregistrement
        crisis_dates_str = [date.strftime('%Y-%m-%d') for date in crisis_dates]
        
        # Sauvegarder les dates de crise dans un fichier pour référence
        with open('crisis_dates.txt', 'w') as f:
            f.write('\n'.join(crisis_dates_str))
        
        print(f"Dates de crise sauvegardées dans 'crisis_dates.txt'")
        
        # Maintenant, exécuter l'analyse principale sur ces dates spécifiques
        results_dict = {}
        selected_factors = {}
        
        for scheme in schemes:
            print(f"\nTraitement du schéma de pondération pour les crises: {scheme}")
            
            # Réinitialiser le sélecteur pour chaque schéma
            crisis_selector = IterativeFactorSelection(
                scheme,
                start_date_full,
                end_date_full,
                region_factors_X='world',
                region_factors_y='US'
            )
            
            # Filtrer les données pour ne garder que les périodes de crise
            crisis_selector.factors_df = crisis_selector.factors_df.loc[crisis_dates]
            crisis_selector.market_return = crisis_selector.market_return.loc[crisis_dates]
            
            # Vérifier si nous avons suffisamment de données
            if len(crisis_selector.factors_df) < 36:  # Minimum de 36 mois pour l'analyse
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

def extension_individual_crisis_data(schemes: list, crisis_name: str, start_date: str, end_date: str):
    """Fonction pour exécuter l'analyse sur une période de crise spécifique."""
    
    try:
        print(f"\nAnalyse de la crise {crisis_name}: {start_date} à {end_date}")
        results_dict, summary_df, factor_table = main(
            weighting_schemes=schemes, 
            start_date=start_date,
            end_date=end_date
        )
        
        if summary_df is not None:
            summary_df['Period'] = crisis_name
        
        return results_dict, summary_df, factor_table
    except Exception as e:
        print(f"Erreur lors de l'analyse de la période {crisis_name}: {str(e)}")
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
    
    # 1. Analyse sur les données de base
    print("\n" + "="*50)
    print("ANALYSE SUR LA PÉRIODE DE BASE")
    print("="*50)
    base_results, base_summary, base_table = extension_base_data(schemes)
    
    if base_summary is not None:
        base_summary['Period'] = 'Base'
    
    # 2. Analyse sur les périodes de crise agrégées
    print("\n" + "="*50)
    print("ANALYSE SUR L'AGRÉGATION DES PÉRIODES DE CRISE")
    print("="*50)
    crisis_results, crisis_summary, crisis_table = extension_all_crisis_data(schemes, crisis_periods)
    
    # 3. Analyses individuelles des périodes de crise
    print("\n" + "="*50)
    print("ANALYSES INDIVIDUELLES DES PÉRIODES DE CRISE")
    print("="*50)
    
    individual_summaries = []
    
    for crisis_name, (start_date, end_date) in crisis_periods.items():
        print("\n" + "-"*30)
        print(f"ANALYSE DE LA CRISE: {crisis_name}")
        print("-"*30)
        
        _, summary, _ = extension_individual_crisis_data(schemes, crisis_name, start_date, end_date)
        
        if summary is not None:
            individual_summaries.append(summary)
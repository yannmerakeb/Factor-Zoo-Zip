#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter l'analyse des facteurs avec PCA/ACP
"""

import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import des modules nécessaires
from extension_ACP_main import main_pca_extension

# Configuration des paramètres
CONFIG = {
    'weighting_schemes': ['VW_cap'],  # Schémas de pondération à analyser
    'start_date': '1993-08-01',       # Date de début de l'analyse
    'end_date': '2021-12-31',         # Date de fin de l'analyse
    'region': 'world',                # Région d'analyse (world, US, etc.)
    'n_components_range': [2, 3, 4], # Nombre de composantes à tester
    'visualization_dir': 'results_pca' # Dossier pour sauvegarder les résultats
}

if __name__ == "__main__":
    print("=" * 80)
    print("EXÉCUTION DE L'ANALYSE FACTOR ZOO AVEC EXTENSION ACP")
    print("=" * 80)
    
    # Création du dossier de résultats s'il n'existe pas
    os.makedirs(CONFIG['visualization_dir'], exist_ok=True)
    
    # Exécution de l'analyse principale
    try:
        results = main_pca_extension(
            weighting_schemes=CONFIG['weighting_schemes'],
            start_date=CONFIG['start_date'],
            end_date=CONFIG['end_date'],
            region=CONFIG['region'],
            n_components_range=CONFIG['n_components_range'],
            visualization_dir=CONFIG['visualization_dir']
        )
        
        print("\nRÉSUMÉ DES RÉSULTATS:")
        for key in results:
            # Modification ici pour gérer correctement le format des clés
            parts = key.split('_')
            scheme = parts[0]
            n_components = parts[-1]  # Prend la dernière partie comme le nombre de composantes
            
            print(f"\n{scheme} avec {n_components} composantes:")
            
            # Afficher la variance expliquée
            if 'pca_results' in results[key]:
                var_exp = results[key]['pca_results']['cum_explained_variance'][-1]
                print(f"  - Variance expliquée: {var_exp:.4f}")
            
            # Afficher les statistiques d'évaluation
            if 'evaluation' in results[key]:
                orig_sig = results[key]['evaluation']
                orig_sig = orig_sig[(orig_sig['type'] == 'Original') & (orig_sig['t_stat'].abs() > 2)].shape[0]
                purif_sig = results[key]['evaluation']
                purif_sig = purif_sig[(purif_sig['type'] == 'Purifié') & (purif_sig['t_stat'].abs() > 2)].shape[0]
                print(f"  - Facteurs originaux significatifs: {orig_sig}")
                print(f"  - Facteurs purifiés significatifs: {purif_sig}")
            
            # Afficher les résultats de sélection
            if 'selection_results' in results[key] and not results[key]['selection_results'].empty:
                top_factors = results[key]['selection_results'].head(3)['factor'].tolist()
                print(f"  - Top 3 facteurs sélectionnés: {', '.join(top_factors)}")
        
        print("\nTous les résultats ont été sauvegardés dans:", CONFIG['visualization_dir'])
        print("\nANALYSE TERMINÉE AVEC SUCCÈS")
        
    except Exception as e:
        print(f"\nERREUR LORS DE L'EXÉCUTION: {e}")
        import traceback
        traceback.print_exc()
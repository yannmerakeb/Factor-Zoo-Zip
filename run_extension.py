from factor_zoo import DataLoader
from ExtensionACP import PCAPurification
import os
import matplotlib.pyplot as plt

# Fermer toutes les figures matplotlib ouvertes
plt.close('all')

# Charger les données
data_loader = DataLoader('VW_cap')
factors_df, market_ret = data_loader.load_factor_data()

# Convertir en décimal si nécessaire
if abs(factors_df.mean().mean()) > 0.05:
    print("Conversion des données de pourcentage en décimal")
    factors_df = factors_df / 100
    market_ret = market_ret / 100

# Initialiser et exécuter l'analyse PCA
pca_extension = PCAPurification(factors_df, market_ret)

# Étapes du PCA
pca_extension.preprocess_data()
pca_results = pca_extension.run_pca(n_components=10)

# Purifier les facteurs
purified_factors = pca_extension.purify_factors(n_components=10)

# Afficher les résultats
print("\nRésultats de l'ACP:")
print(f"Nombre de composantes: {pca_results['n_components']}")
print(f"Variance expliquée totale: {pca_results['cum_explained_variance'][-1]:.4f}")

# Créer un dossier pour les résultats
output_dir = "results_pca"
os.makedirs(output_dir, exist_ok=True)

# Dictionnaire des figures à sauvegarder
figures_to_save = {
    'scree_plot': pca_extension.plot_scree(n_components=10),
    'loadings': pca_extension.plot_factor_loadings(n_components=5),
    'dendrogram': pca_extension.plot_factor_dendrogram(),
    'comparison': pca_extension.compare_original_vs_purified(),
}

# Évaluation des facteurs purifiés
results_df, fig_eval = pca_extension.evaluate_purified_factors(market_return=market_ret)
figures_to_save['evaluation'] = fig_eval

# Sauvegarder toutes les figures
for name, fig in figures_to_save.items():
    filename = os.path.join(output_dir, f'{name}.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Fermer la figure pour libérer la mémoire
    print(f"Sauvegardé: {filename}")

print(f"\nAnalyse PCA terminée! Tous les graphiques ont été sauvegardés dans '{output_dir}'.")
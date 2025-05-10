from factor_zoo import DataLoader
from ExtensionACP import PCAPurification

# Charger les données
data_path = "data/[usa]_[all_factors]_[monthly]_[vw_cap].csv"
dl = DataLoader(data_path)
factors_df, market_ret = dl.load_factor_data()

# Convertir en décimal si nécessaire
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
print(pca_results)
print(purified_factors.head())

# Visualisations (optionnelles mais recommandées pour vérification)
fig_scree = pca_extension.plot_scree(n_components=10)
fig_scree.show()

fig_loadings = pca_extension.plot_factor_loadings(n_components=5)
fig_loadings.show()

fig_dendrogram = pca_extension.plot_factor_dendrogram()
fig_dendrogram.show()

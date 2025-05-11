# run_temporal_extension.py
from data_loader import DataLoader
from ExtensionTemporelleAvancee import run_temporal_analysis

# Charger les données
data_loader = DataLoader('VW_cap', '1993-08-01', '2021-12-31')
factors_df, market_ret = data_loader.load_factor_data('US')

# Convertir en décimal si nécessaire
if abs(market_ret.mean()) > 0.05:
    factors_df = factors_df / 100
    market_ret = market_ret / 100

# Exécuter l'analyse temporelle
results = run_temporal_analysis(factors_df, market_ret)

# Accéder aux résultats disponibles
adaptive_selection = results['adaptive_selection']
temporal_stability = results['temporal_stability']
regime_analysis = results['regime_analysis']
temporal_obj = results['temporal_object']

# Afficher des informations sur l'évolution temporelle
print("\n=== Analyse de l'évolution temporelle ===")
print(f"Période analysée: {adaptive_selection['date'].min()} à {adaptive_selection['date'].max()}")
print(f"Nombre de fenêtres: {len(adaptive_selection)}")

# Évolution du nombre de facteurs
print("\nÉvolution du nombre de facteurs sélectionnés:")
print(f"- Minimum: {adaptive_selection['n_factors'].min()}")
print(f"- Maximum: {adaptive_selection['n_factors'].max()}")
print(f"- Moyenne: {adaptive_selection['n_factors'].mean():.1f}")
print(f"- Écart-type: {adaptive_selection['n_factors'].std():.1f}")

# Tendance temporelle
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(
    range(len(adaptive_selection)), 
    adaptive_selection['n_factors']
)
print(f"- Tendance: {'décroissante' if slope < 0 else 'croissante'} (pente: {slope:.3f}, p-value: {p_value:.3f})")

# Identifier les facteurs toujours présents vs jamais présents
always_selected = temporal_stability[temporal_stability['selection_frequency'] > 0.9]
never_selected = temporal_stability[temporal_stability['selection_frequency'] == 0]

print(f"\nFacteurs (presque) toujours sélectionnés ({len(always_selected)}):")
if len(always_selected) > 0:
    for factor in always_selected.index:
        print(f"- {factor}: {always_selected.loc[factor, 'selection_frequency']:.1%}")
else:
    print("- Aucun facteur n'est sélectionné dans plus de 90% des fenêtres")

print(f"\nNombre de facteurs jamais sélectionnés: {len(never_selected)}")

# Analyser les régimes de volatilité
print("\n=== Analyse des régimes (High Vol vs Low Vol) ===")
high_vol_df = regime_analysis['High_Vol']
low_vol_df = regime_analysis['Low_Vol']

# Facteurs performant mieux selon le régime
high_vol_better = []
low_vol_better = []

for factor in factors_df.columns:
    high_vol_row = high_vol_df[high_vol_df['factor'] == factor]
    low_vol_row = low_vol_df[low_vol_df['factor'] == factor]
    
    if not high_vol_row.empty and not low_vol_row.empty:
        high_vol_sharpe = high_vol_row['sharpe'].values[0]
        low_vol_sharpe = low_vol_row['sharpe'].values[0]
        
        if high_vol_sharpe > low_vol_sharpe:
            high_vol_better.append((factor, high_vol_sharpe - low_vol_sharpe))
        else:
            low_vol_better.append((factor, low_vol_sharpe - high_vol_sharpe))

print("\nTop 5 facteurs performant mieux en haute volatilité:")
high_vol_better.sort(key=lambda x: x[1], reverse=True)
for factor, diff in high_vol_better[:5]:
    print(f"- {factor}: +{diff:.3f} Sharpe")

print("\nTop 5 facteurs performant mieux en basse volatilité:")
low_vol_better.sort(key=lambda x: x[1], reverse=True)
for factor, diff in low_vol_better[:5]:
    print(f"- {factor}: +{diff:.3f} Sharpe")

# Identifier les changements majeurs dans la sélection
print("\n=== Changements majeurs dans la sélection ===")
major_changes = []
for i in range(1, len(adaptive_selection)):
    prev_factors = set(adaptive_selection.iloc[i-1]['factors'])
    curr_factors = set(adaptive_selection.iloc[i]['factors'])
    
    added = curr_factors - prev_factors
    removed = prev_factors - curr_factors
    
    if len(added) + len(removed) > 3:  # Changement majeur (seuil abaissé à 3)
        date = adaptive_selection.iloc[i]['date']
        major_changes.append({
            'date': date,
            'added': list(added),
            'removed': list(removed),
            'total_change': len(added) + len(removed)
        })

if major_changes:
    for change in major_changes[:5]:  # Afficher les 5 premiers
        print(f"\n{change['date'].strftime('%Y-%m')}:")
        if change['added']:
            print(f"  Ajoutés: {change['added']}")
        if change['removed']:
            print(f"  Retirés: {change['removed']}")
else:
    print("Aucun changement majeur détecté")

# Analyse de la persistance par cluster
try:
    from clusters import create_factor_clusters
    get_cluster = create_factor_clusters()
    
    # Ajouter les clusters au DataFrame de stabilité
    temporal_stability['cluster'] = temporal_stability.index.map(get_cluster)
    
    # Statistiques par cluster
    cluster_stats = temporal_stability.groupby('cluster').agg({
        'selection_frequency': ['mean', 'max', 'count'],
        'avg_streak': 'mean'
    })
    
    print("\n=== Analyse par cluster ===")
    print(cluster_stats.round(3))
    
    # Cluster le plus stable
    most_stable_cluster = cluster_stats['selection_frequency']['mean'].idxmax()
    print(f"\nCluster le plus stable: {most_stable_cluster}")
    
except ImportError:
    print("\nAnalyse par cluster non disponible (clusters.py manquant)")

# Analyser l'évolution de l'alpha moyen
print("\n=== Évolution de l'alpha moyen ===")
alpha_evolution = adaptive_selection['avg_alpha'].dropna()
if len(alpha_evolution) > 0:
    print(f"Alpha moyen au début: {alpha_evolution.iloc[0]:.2f}%")
    print(f"Alpha moyen à la fin: {alpha_evolution.iloc[-1]:.2f}%")
    
    # Tendance de l'alpha
    slope_alpha, intercept_alpha, r_value_alpha, p_value_alpha, std_err_alpha = stats.linregress(
        range(len(alpha_evolution)), 
        alpha_evolution
    )
    print(f"Tendance de l'alpha: {'décroissante' if slope_alpha < 0 else 'croissante'} (pente: {slope_alpha:.3f}, p-value: {p_value_alpha:.3f})")

# Sauvegarder les résultats dans des fichiers CSV
adaptive_selection.to_csv('temporal_adaptive_selection.csv', index=False)
temporal_stability.to_csv('temporal_stability.csv')

print("\n=== Résultats sauvegardés ===")
print("- temporal_adaptive_selection.csv")
print("- temporal_stability.csv")
print("- temporal_adaptive_selection.png")
print("- temporal_stability.png")

# Afficher un résumé final
print("\n=== Résumé final ===")
print(f"1. Le nombre de facteurs nécessaires a évolué de {adaptive_selection['n_factors'].min()} à {adaptive_selection['n_factors'].max()}")
print(f"2. En moyenne, {adaptive_selection['n_factors'].mean():.1f} facteurs sont sélectionnés par fenêtre")
print(f"3. Les facteurs les plus stables sont: {', '.join(temporal_stability.nlargest(3, 'selection_frequency').index)}")
print(f"4. {len(never_selected)} facteurs ne sont jamais sélectionnés sur la période")

# Identifier les périodes clés
min_factors_idx = adaptive_selection['n_factors'].idxmin()
max_factors_idx = adaptive_selection['n_factors'].idxmax()
print(f"\n5. Période avec le moins de facteurs: {adaptive_selection.loc[min_factors_idx, 'date'].strftime('%Y-%m')} ({adaptive_selection.loc[min_factors_idx, 'n_factors']} facteurs)")
print(f"6. Période avec le plus de facteurs: {adaptive_selection.loc[max_factors_idx, 'date'].strftime('%Y-%m')} ({adaptive_selection.loc[max_factors_idx, 'n_factors']} facteurs)")
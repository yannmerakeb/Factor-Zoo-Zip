# data_loader_fixed.py
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Chargement et nettoyage des données de facteurs."""
    def __init__(self, weighting, start_date="1971-11-01", end_date="2021-12-31"):
        self.data_path = f'data/{weighting}.csv'
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def load_factor_data(self, region='world'):
        """Charge les données des facteurs et extrait le rendement du marché."""
        print(f"Chargement des données depuis {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        # Filtrer par région
        if region == 'US':
            df = df[df['location'] == 'usa']
        elif region == 'ex US' or region == 'World_ex_US':
            df = df[df['location'] != 'usa']
        # Pour 'world' ou 'World', on garde toutes les régions

        # Pondération des facteurs par le nombre de stocks
        if region in ['ex US', 'World_ex_US', 'world', 'World']:
            # Supprimer les duplicatas potentiels
            df = df.drop_duplicates(subset=['date', 'name'], keep='first')
            
            # Calculer la moyenne pondérée par nombre de stocks
            df['weighted_ret'] = df['ret'] * df['n_stocks']
            weighted_sum = df.groupby(['date', 'name'])['weighted_ret'].sum()
            stock_count = df.groupby(['date', 'name'])['n_stocks'].sum()
            avg_returns = weighted_sum / stock_count
            
            # Créer le pivot DataFrame
            pivot_df = avg_returns.unstack()
            
        else:  # Pour 'US' seulement
            # Pour US, pas de duplicatas donc on peut directement pivoter
            pivot_df = df.pivot(index='date', columns='name', values='ret')

        # Extraire le rendement du marché
        if 'market_equity' in pivot_df.columns:
            market_return = pivot_df['market_equity']
            factors_df = pivot_df.drop(columns=['market_equity'])
        else:
            # Si market_equity n'existe pas, utiliser la moyenne comme proxy
            market_return = pivot_df.mean(axis=1)
            factors_df = pivot_df

        print(f"Données chargées: {len(factors_df)} périodes (mois), {factors_df.shape[1]} facteurs")
        
        # S'assurer que les indices sont de type datetime
        if not isinstance(factors_df.index, pd.DatetimeIndex):
            factors_df.index = pd.to_datetime(factors_df.index)
        if not isinstance(market_return.index, pd.DatetimeIndex):
            market_return.index = pd.to_datetime(market_return.index)
            
        return factors_df, market_return
    
    def check_factors(self):
        """Vérifie les facteurs disponibles"""
        df = pd.read_csv(self.data_path)
        
        # Facteurs uniques
        unique_factors = df['name'].unique()
        print(f"Nombre total de facteurs (incluant market_equity): {len(unique_factors)}")
        
        # Sans market_equity
        non_market_factors = [f for f in unique_factors if f != 'market_equity']
        print(f"Nombre de facteurs sans market_equity: {len(non_market_factors)}")
        
        # Afficher quelques facteurs pour vérifier
        print("\nPremiers 10 facteurs (alphabétique):")
        for f in sorted(unique_factors)[:10]:
            print(f"  - {f}")
        
        # Vérifier si RMRF est présent
        print(f"\nRMRF présent: {'RMRF' in unique_factors}")
        print(f"market_equity présent: {'market_equity' in unique_factors}")
        
        # Vérifier les régions disponibles
        if 'location' in df.columns:
            locations = df['location'].unique()
            print(f"\nRégions disponibles: {locations}")
            
            # Compter les observations par région
            location_counts = df.groupby('location').size()
            print("\nObservations par région:")
            for loc, count in location_counts.items():
                print(f"  - {loc}: {count}")
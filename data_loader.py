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
        
    def load_factor_data(self, region = 'world'):
        """Charge les données des facteurs et extrait le rendement du marché."""
        print(f"Chargement des données depuis {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        if region == 'US':
            df = df[df['location'] == 'usa']
        elif region == 'ex US':
            df = df[df['location'] != 'usa']

        # Pondération des facteurs par le nombre de stocks
        if region in ['ex US', 'world']:
            df['weighted_ret'] = df['ret'] * df['n_stocks']
            weighted_sum = df.groupby(['date', 'name'])['weighted_ret'].sum()
            stock_count = df.groupby(['date', 'name'])['n_stocks'].sum()
            pivot_df = (weighted_sum / stock_count).unstack()

        else:
            pivot_df = df.pivot(index='date', columns='name', values='ret')

        market_return = pivot_df['market_equity']
        factors_df = pivot_df.drop(columns=['market_equity'])

        print(f"Données chargées: {len(factors_df)} périodes (mois), {factors_df.shape[1]} facteurs")
        return factors_df, market_return
    
    def diagnostic_check(self, factors_df, market_return):
        """Vérifie les données et suggère des corrections."""
        
        print("DIAGNOSTIC DES DONNÉES")
        print("=" * 50)
        
        # Vérifier le marché
        print(f"Marché - Moyenne: {market_return.mean():.6f}")
        print(f"Marché - Écart-type: {market_return.std():.6f}")
        
        # Vérifier quelques facteurs clés
        key_factors = ['cop_at', 'noa_gr1a', 'saleq_gr1', 'ival_me', 'resff3_12_1']
        
        for factor in key_factors:
            if factor in factors_df.columns:
                print(f"\n{factor}:")
                print(f"  Moyenne: {factors_df[factor].mean():.6f}")
                print(f"  Écart-type: {factors_df[factor].std():.6f}")
                
                # Test de régression simple
                y = factors_df[factor]
                X = sm.add_constant(market_return)
                valid_idx = ~(y.isna() | market_return.isna())
                
                model = sm.OLS(y[valid_idx], X[valid_idx])
                results = model.fit()
                
                print(f"  Alpha CAPM: {results.params[0]:.6f}")
                print(f"  t-stat: {results.tvalues[0]:.3f}")
                print(f"  Alpha annualisé: {results.params[0] * 12 * 100:.2f}%")
        
        return {
            'market_mean': market_return.mean(),
            'market_std': market_return.std(),
            'likely_percentage': abs(market_return.mean()) > 0.05
        }
    
    def check_factors(self):
        df = pd.read_csv(self.ata_path)
        
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


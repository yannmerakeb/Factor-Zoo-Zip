# test_factors.py
import pandas as pd

def check_factors():
    data_path = "data/[usa]_[all_factors]_[monthly]_[vw_cap].csv"
    df = pd.read_csv(data_path)
    
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

if __name__ == "__main__":
    check_factors()
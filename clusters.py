def create_factor_clusters():
    """
    Crée un dictionnaire associant chaque facteur à son cluster en se basant sur l'Exhibit 3 de l'article.
    Ces clusters sont utilisés pour la visualisation des alphas dans l'Exhibit 4.
    """
    clusters = {
        # Market
        'RMRF': 'Market',
        'market_equity': 'Market',
        
        # Quality
        'cop_at': 'Quality',
        'ni_at': 'Quality',
        'niq_at': 'Quality',
        
        # Investment
        'noa_gr1a': 'Investment',
        'saleq_gr1': 'Investment',
        'rnoa_gr1a': 'Investment',
        'nncoa_gr1a': 'Investment',
        
        # Value
        'ival_me': 'Value',
        'debt_me': 'Value',
        'ocf_me': 'Value',
        'ni_me': 'Value',
        'dsale_dinv': 'Value',
        'at_me': 'Value',
        
        # Momentum
        'resff3_12_1': 'Momentum',
        'ret_12_1': 'Momentum',
        
        # Seasonality
        'seas_6_10an': 'Seasonality',
        'seas_11_15na': 'Seasonality',
        'seas_16_20an': 'Seasonality',
        
        # Low Risk
        'seas_6_10na': 'Low Risk',
        'zero_trades_252d': 'Low Risk',
        'zero_trades_21d': 'Low Risk',
        'turnover_126d': 'Low Risk',
        'ivol_ff3_21d': 'Low Risk',
        
        # Profitability
        'o_score': 'Profitability',
        'ni_be': 'Profitability',
        
        # Debt Issuance
        'ni_ar1': 'Debt Issuance',
        'noa_at': 'Debt Issuance',
        'nfna_gr1a': 'Debt Issuance',
        
        # Low Leverage
        'age': 'Low Leverage',
        'aliq_mat': 'Low Leverage',
        
        # Profit Growth
        'dsale_dinv': 'Profit Growth',
        
        # Short-Term Reversal
        'rmax5_rvol_21d': 'Short-Term Reversal',
        
        # Accruals
        'cowc_gr1a': 'Accruals',
        
        # Size
        'size': 'Size'
    }
    
    # Fonction pour attribuer un cluster par défaut à un facteur non répertorié
    def get_cluster(factor):
        # Si le facteur est déjà dans le dictionnaire, retourner son cluster
        if factor in clusters:
            return clusters[factor]
        
        # Sinon, essayer de déduire le cluster à partir du nom du facteur
        if 'gr' in factor and ('noa' in factor or 'coa' in factor):
            return 'Investment'
        elif 'me' in factor or 'be' in factor:
            return 'Value'
        elif 'seas' in factor:
            return 'Seasonality'
        elif 'ret' in factor or 'mom' in factor or 'res' in factor:
            return 'Momentum'
        elif 'zero' in factor or 'trades' in factor or 'vol' in factor:
            return 'Low Risk'
        elif 'cop' in factor or 'ni_' in factor or 'roa' in factor:
            return 'Quality'
        elif 'debt' in factor or 'ar' in factor:
            return 'Debt Issuance'
        elif 'lev' in factor or 'age' in factor:
            return 'Low Leverage'
        elif 'sale' in factor and 'gr' in factor:
            return 'Profit Growth'
        elif 'acc' in factor or 'cowc' in factor:
            return 'Accruals'
        elif 'size' in factor or 'market' in factor:
            return 'Size'
        else:
            # Catégorie par défaut
            return 'Other'
    
    # Cette fonction permet d'obtenir le cluster d'un facteur, même s'il n'est pas explicitement défini
    return get_cluster
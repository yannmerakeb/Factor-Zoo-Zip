def create_factor_clusters():
    """
    Crée un dictionnaire associant chaque facteur à son cluster en se basant sur le graphique des alphas.
    Ces clusters sont représentés par différentes couleurs dans la visualisation.
    """
    
    # Dictionnaire complet des clusters basé sur le graphique
    clusters = {
        # Premier groupe (turquoise)
        'netdebt_me': 'Debt/Price',
        'be_me': 'Debt/Price',
        'currat': 'Debt/Price',
        'cash_at': 'Debt/Price',
        'debt_at': 'Debt/Price',
        'niq_be': 'Debt/Price',
        
        # Deuxième groupe (orange)
        'debt_me': 'Value',
        'fcf_me': 'Value',
        'ocf_me': 'Value',
        'ebit_me': 'Value',
        'ni_me': 'Value',
        'cfo_me': 'Value',
        
        # Troisième groupe (rose/violet)
        'at_be': 'Profitability',
        'niq_at': 'Profitability',
        'cop_at': 'Profitability',
        'ope_be': 'Profitability',
        'gp_at': 'Profitability',
        'ni_at': 'Profitability',
        'ope_at': 'Profitability',
        'ebit_at': 'Profitability',
        'sale_at': 'Profitability',
        'gp_sale': 'Profitability',
        'cfroi': 'Profitability',
        'grprof': 'Profitability',
        'grltnoa': 'Profitability',
        'oiadp_at': 'Profitability',
        'oaccruals_at': 'Profitability',
        'taccruals_at': 'Profitability',
        'eq_dur': 'Profitability',
        'eq_offer': 'Profitability',
        'oaccruals_ni': 'Profitability',
        
        # Quatrième groupe (bleu foncé/négatif)
        'asset_gr': 'Growth',
        'fnl_gr1': 'Growth',
        'inv_gr1': 'Growth',
        'capx_gr1': 'Growth',
        'capx_gr2': 'Growth',
        'capx_gr3': 'Growth',
        'fnl_gr2': 'Growth',
        'emp_gr1': 'Growth',
        'inv_gr2': 'Growth',
        'coa_gr1a': 'Growth',
        'noa_gr1a': 'Growth',
        'ppeinv_gr1a': 'Growth',
        'nncoa_gr1a': 'Growth',
        'lnoa_gr1a': 'Growth',
        'cowc_gr1a': 'Growth',
        
        # Cinquième groupe (vert clair)
        'ret_6_1': 'Momentum',
        'ret_9_1': 'Momentum',
        'ret_12_1': 'Momentum',
        'ret_12_6': 'Momentum',
        'resff3_6_1': 'Momentum',
        'resff3_9_1': 'Momentum',
        'resff3_12_1': 'Momentum',
        'resff3_12_6': 'Momentum',
        'ret_1_0': 'Momentum',
        'resff3_1_0': 'Momentum',
        'ret_60_12': 'Momentum',
        'resff3_60_12': 'Momentum',
        'bbscore': 'Momentum',
        'bbkind': 'Momentum',
        'bbrdg1': 'Momentum',
        'bbrdg2': 'Momentum',
        'bbrgrd': 'Momentum',
        'bbrgr': 'Momentum',
        'bbrshr': 'Momentum',
        
        # Sixième groupe (bleu clair)
        'idiovol': 'Volatility',
        'rvol_21d': 'Volatility',
        'ivol_ff3_21d': 'Volatility',
        'iskew_ff3_21d': 'Volatility',
        'ivol_capm_252d': 'Volatility',
        'iskew_capm_21d': 'Volatility',
        'betabab_1260d': 'Volatility',
        'beta_60m': 'Volatility',
        
        # Septième groupe (vert foncé)
        'rmax1_21d': 'Liquidity',
        'rmax5_21d': 'Liquidity',
        'rmax5_rvol_21d': 'Liquidity',
        'ami_126d': 'Liquidity',
        'zero_trades_252d': 'Liquidity',
        'turnover_var_126d': 'Liquidity',
        
        # Huitième groupe (marron)
        'age': 'Miscellaneous1',
        'size': 'Miscellaneous1',
        'aliq_at': 'Miscellaneous1',
        'aliq_mat': 'Miscellaneous1',
        'at_me': 'Miscellaneous1',
        'seas_1_1an': 'Miscellaneous1',
        'cash_me': 'Miscellaneous1',
        'chcsho_12m': 'Miscellaneous1',
        
        # Neuvième groupe (bordeaux)
        'ni_inc8q': 'Earnings',
        'ni_ar1': 'Earnings',
        'ni_ar8q': 'Earnings',
        'dsale_dinv': 'Earnings',
        'niq_su': 'Earnings',
        'niq_be_chg1': 'Earnings',
        'saleq_su': 'Earnings',
        'stdcf_12m': 'Earnings',
        'seas_6_10an': 'Earnings',
        'seas_11_15an': 'Earnings',
        'seas_16_20an': 'Earnings',
        
        # Dixième groupe (bleu marine)
        'ni_chg1': 'Growth Trends',
        'ni_chg1_8q': 'Growth Trends',
        'niq_gr1': 'Growth Trends',
        'ebit_gr1': 'Growth Trends',
        'sale_gr1': 'Growth Trends',
        'sale_gr3': 'Growth Trends',
        'gp_gr1': 'Growth Trends',
        'ope_gr1': 'Growth Trends',
        'saleq_gr1': 'Growth Trends',
        'seas_2_5an': 'Growth Trends',
        
        # Onzième groupe (vert olive)
        'o_score': 'Financial Health',
        'z_score': 'Financial Health',
        'zscore': 'Financial Health',
        'o_score_fen': 'Financial Health',
        'pctacc': 'Financial Health',
        
        # Douzième groupe (turquoise clair avec négatifs)
        'rd_me': 'Research',
        'rd_sale': 'Research',
        'adv_sale': 'Research',
        'rd_mve': 'Research',
        
        # Treizième groupe (bleu)
        'xinst_perc': 'Institutional',
        'xrd_var5': 'Institutional',
        'xinst_mean': 'Institutional',
        'xinst_std': 'Institutional',
        'chmom': 'Institutional',
        'nanalyst': 'Institutional',
        'chtx': 'Institutional',
        'beta_dimson_21d': 'Institutional',
        'ear_vol': 'Institutional',
        'div12m_me': 'Institutional',
        'ami_21d': 'Institutional',
        'dolvol_21d': 'Institutional',
        'dolvol_252d': 'Institutional',
        'dolvol_var_126d': 'Institutional',
        'ill_rvol_21d': 'Institutional',
        'issue_perc': 'Institutional',
        'kz_index': 'Institutional',
        'market_equity': 'Institutional',
        'mve_ia': 'Institutional',
        'prc': 'Institutional',
        'retvol': 'Institutional',
        'rvol_var_21d': 'Institutional',
        'turn_126d': 'Institutional',
        'turn_252d': 'Institutional',
        'turnover_126d': 'Institutional',
        'turnover_252d': 'Institutional',
        
        # Facteurs de référence
        'RMRF': 'Market Factors',
        'market_equity': 'Market Factors',
    }
    
    # Fonction pour attribuer un cluster par défaut à un facteur non répertorié
    def get_cluster(factor):
        # Si le facteur est déjà dans le dictionnaire, retourner son cluster
        if factor in clusters:
            return clusters[factor]
        
        # Sinon, essayer de déduire le cluster à partir du nom du facteur
        if 'gr' in factor and ('noa' in factor or 'coa' in factor or 'inv' in factor or 'capx' in factor):
            return 'Growth'
        elif 'me' in factor and ('at' not in factor):
            return 'Value'
        elif 'seas' in factor:
            return 'Earnings'
        elif 'ret' in factor or 'mom' in factor or 'res' in factor:
            return 'Momentum'
        elif 'vol' in factor or 'beta' in factor:
            return 'Volatility'
        elif 'zero' in factor or 'trades' in factor or 'turn' in factor:
            return 'Liquidity'
        elif 'cop' in factor or 'ni_at' in factor or 'roa' in factor or 'gp' in factor:
            return 'Profitability'
        elif 'debt' in factor or 'cash' in factor:
            return 'Debt/Price'
        elif 'size' in factor or 'market' in factor:
            return 'Institutional'
        elif 'chg' in factor or 'gr' in factor:
            return 'Growth Trends'
        elif 'o_score' in factor or 'z_score' in factor:
            return 'Financial Health'
        elif 'rd' in factor:
            return 'Research'
        else:
            # Catégorie par défaut
            return 'Other'
    
    # Cette fonction permet d'obtenir le cluster d'un facteur, même s'il n'est pas explicitement défini
    return get_cluster
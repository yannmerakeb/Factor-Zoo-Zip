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


cluster_mapping = {
    # Accruals
    "cowc_gr1a": "Accruals",
    "oaccruals_at": "Accruals",
    "oaccruals_ni": "Accruals",
    "eas 16 20na": "Accruals",
    "taccruals_at": "Accruals",
    "taccruals_ni": "Accruals",

    # Debt Issuance
    "capex_abn": "Debt Issuance",
    "debt_gr3": "Debt Issuance",
    "fnl_gr1a": "Debt Issuance",
    "ncol_gr1a": "Debt Issuance",
    "nfna_gr1a": "Debt Issuance",
    "ni_ar1": "Debt Issuance",
    "noa_at": "Debt Issuance",

    # Investment
    "aliq_at": "Investment",
    "at_gr1": "Investment",
    "be_gr1a": "Investment",
    "capx_gr1": "Investment",
    "capx_gr2": "Investment",
    "capx_gr3": "Investment",
    "coa_gr1a": "Investment",
    "col_gr1a": "Investment",
    "emp_gr1": "Investment",
    "inv_gr1": "Investment",
    "inv_gr1a": "Investment",
    "lnoa_gr1a": "Investment",
    "mispricing_mgmt": "Investment",
    "ncoa_gr1a": "Investment",
    "nncoa_gr1a": "Investment",
    "noa_gr1a": "Investment",
    "ppeinv_gr1a": "Investment",
    "ret_60_12": "Investment",
    "sale_gr1": "Investment",
    "sale_gr3": "Investment",
    "saleq_gr1": "Investment",
    "seas_2_5na": "Investment",

    # Low Leverage
    "age": "Low Leverage",
    "aliq_mat": "Low Leverage",
    "at_be": "Low Leverage",
    "bidaskhl_21d": "Low Leverage",
    "cash_at": "Low Leverage",
    "netdebt_me": "Low Leverage",
    "ni_ivol": "Low Leverage",
    "rd_sale": "Low Leverage",
    "rd5_at": "Low Leverage",
    "tangibility": "Low Leverage",
    "z_score": "Low Leverage",

    # Low Risk
    "beta_60m": "Low Risk",
    "beta_dimson_21d": "Low Risk",
    "betabab_1260d": "Low Risk",
    "betadown_252d": "Low Risk",
    "earnings_variability": "Low Risk",
    "ivol_capm_21d": "Low Risk",
    "ivol_capm_252d": "Low Risk",
    "ivol_ff3_21d": "Low Risk",
    "ivol_hxz4_21d": "Low Risk",
    "ocfq_saleq_std": "Low Risk",
    "rmax1_21d": "Low Risk",
    "rmax5_21d": "Low Risk",
    "rvol_21d": "Low Risk",
    "seas_6_10na": "Low Risk",
    "turnover_126d": "Low Risk",
    "zero_trades_21d": "Low Risk",
    "zero_trades_126d": "Low Risk",
    "zero_trades_252d": "Low Risk",

    # Momentum
    "prc_highprc_252d": "Momentum",
    "resff3_6_1": "Momentum",
    "resff3_12_1": "Momentum",
    "ret_3_1": "Momentum",
    "ret_6_1": "Momentum",
    "ret_9_1": "Momentum",
    "ret_12_1": "Momentum",
    "seas_1_1na": "Momentum",

    # Profit Growth
    "dsale_dinv": "Profit Growth",
    "dsale_drec": "Profit Growth",
    "dsale_dsga": "Profit Growth",
    "niq_at_chg1": "Profit Growth",
    "niq_be_chg1": "Profit Growth",
    "niq_su": "Profit Growth",
    "ocf_at_chg1": "Profit Growth",
    "ret_12_7": "Profit Growth",
    "sale_emp_gr1": "Profit Growth",
    "saleq_su": "Profit Growth",
    "seas_1_1an": "Profit Growth",
    "tax_gr1a": "Profit Growth",

    # Profitability
    "dolvol_var_126d": "Profitability",
    "ebit_bev": "Profitability",
    "ebit_sale": "Profitability",
    "f_score": "Profitability",
    "ni_be": "Profitability",
    "niq_be": "Profitability",
    "o_score": "Profitability",
    "ocf_at": "Profitability",
    "ope_be": "Profitability",
    "ope_bel1": "Profitability",
    "turnover_var_126d": "Profitability",

    # Quality
    "at_turnover": "Quality",
    "cop_at": "Quality",
    "cop_atl1": "Quality",
    "dgp_dsale": "Quality",
    "gp_at": "Quality",
    "gp_atl1": "Quality",
    "mispricing_perf": "Quality",
    "ni_inc8q": "Quality",
    "niq_at": "Quality",
    "op_at": "Quality",
    "op_atl1": "Quality",
    "opex_at": "Quality",
    "qmj": "Quality",
    "qmj_growth": "Quality",
    "qmj_prof": "Quality",
    "qmj_safety": "Quality",
    "sale_bev": "Quality",

    # Seasonality
    "corr_1260d": "Seasonality",
    "coskew_21d": "Seasonality",
    "dbnetis_at": "Seasonality",
    "kz_index": "Seasonality",
    "lti_gr1a": "Seasonality",
    "pi_nix": "Seasonality",
    "seas_2_5an": "Seasonality",
    "seas_6_10an": "Seasonality",
    "seas_11_15an": "Seasonality",
    "seas_11_15na": "Seasonality",
    "seas_16_20an": "Seasonality",
    "sti_gr1a": "Seasonality",

    # Size
    "ami_126d": "Size",
    "dolvol_126d": "Size",
    "market_equity": "Size",
    "prc": "Size",
    "rd_me": "Size",

    # Short-Term Reversal
    "iskew_capm_21d": "Short-Term Reversal",
    "iskew_ff3_21d": "Short-Term Reversal",
    "iskew_hxz4_21d": "Short-Term Reversal",
    "ret_1_0": "Short-Term Reversal",
    "rmax5_rvol_21d": "Short-Term Reversal",
    "rskew_21d": "Short-Term Reversal",

    # Value
    "at_me": "Value",
    "be_me": "Value",
    "bev_mev": "Value",
    "chcsho_12m": "Value",
    "debt_me": "Value",
    "div12m_me": "Value",
    "ebitda_mev": "Value",
    "eq_dur": "Value",
    "eqnetis_at": "Value",
    "eqnpo_12m": "Value",
    "eqnpo_me": "Value",
    "eqpo_me": "Value",
    "fcf_me": "Value",
    "ival_me": "Value",
    "netis_at": "Value",
    "ni_me": "Value",
    "ocf_me": "Value",
    "sale_me": "Value",
}
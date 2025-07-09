# first line: 171
@memory.cache
def process_features_parallel(data_dict, train_start_year, train_end_year, n_cores):
    """
    并行处理特征工程步骤
    Process feature engineering steps in parallel
    
    Args:
        data_dict: 数据字典 (Data dictionary)
        train_start_year: 训练开始年份 (Training start year)
        train_end_year: 训练结束年份 (Training end year)
        n_cores: 使用的CPU核心数 (Number of CPU cores to use)
        
    Returns:
        dict: 处理后的特征字典 (Processed features dictionary)
    """
    print_section("并行处理特征数据 (Processing Features in Parallel)", width=60)
    
    # 男子比赛数据处理 (Men's data processing)
    print("处理男子比赛数据... (Processing men's data...)")
    m_seed_features = process_seeds(
        data_dict['m_tourney_seeds'],
        train_start_year, 
        train_end_year
    )
    
    m_team_stats = calculate_team_stats(
        data_dict['m_regular_season'], 
        data_dict['m_regular_detail'],
        train_start_year, 
        train_end_year
    )
    
    # 并行创建男子对战历史 (Create men's matchup history in parallel)
    print("创建男子对战历史... (Creating men's matchup history...)")
    seasons = list(range(train_start_year, train_end_year + 1))
    
    process_season = partial(
        create_matchup_history_for_season,
        regular_season=data_dict['m_regular_season'],
        tourney_results=data_dict['m_tourney_results']
    )
    
    results = Parallel(n_jobs=n_cores)(
        delayed(process_season)(season) for season in seasons
    )
    
    m_matchup_history = pd.concat(results)
    
    m_progression_probs = calculate_progression_probabilities(
        m_seed_features,
        m_team_stats
    )
    
    # 打包男子特征 (Package men's features)
    m_features_dict = package_features(
        m_team_stats, 
        m_seed_features, 
        m_matchup_history, 
        m_progression_probs
    )
    
    # 新增：女子比赛数据处理 (Women's data processing)
    print("\n处理女子比赛数据... (Processing women's data...)")
    w_seed_features = process_seeds(
        data_dict['w_tourney_seeds'],
        train_start_year, 
        train_end_year
    )
    
    w_team_stats = calculate_team_stats(
        data_dict['w_regular_season'], 
        data_dict['w_regular_detail'],
        train_start_year, 
        train_end_year
    )
    
    # 并行创建女子对战历史 (Create women's matchup history in parallel)
    print("创建女子对战历史... (Creating women's matchup history...)")
    process_w_season = partial(
        create_matchup_history_for_season,
        regular_season=data_dict['w_regular_season'],
        tourney_results=data_dict['w_tourney_results']
    )
    
    w_results = Parallel(n_jobs=n_cores)(
        delayed(process_w_season)(season) for season in seasons
    )
    
    w_matchup_history = pd.concat(w_results)
    
    w_progression_probs = calculate_progression_probabilities(
        w_seed_features,
        w_team_stats
    )
    
    # 打包女子特征 (Package women's features)
    w_features_dict = package_features(
        w_team_stats, 
        w_seed_features, 
        w_matchup_history, 
        w_progression_probs
    )
    
    # 添加538评级特征
    features_dict = integrate_538_ratings(features_dict, data_dict, train_start_year, train_end_year)
    
    # 返回男女特征字典 (Return men's and women's feature dictionaries)
    return {
        'men': m_features_dict,
        'women': w_features_dict
    }

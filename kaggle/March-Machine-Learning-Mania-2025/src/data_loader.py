def load_538_ratings(data_path, use_cache=True, cache_dir=None):
    """
    加载538Rating数据文件并进行预处理
    
    参数:
        data_path (str): 数据文件所在目录
        use_cache (bool): 是否使用缓存
        cache_dir (str): 缓存目录
        
    返回:
        pd.DataFrame: 预处理后的538评级数据
    """
    import os
    import pandas as pd
    import numpy as np
    
    # 文件路径
    ratings_file = os.path.join(data_path, '538Rating.csv')
    
    # 检查文件是否存在
    if not os.path.exists(ratings_file):
        print(f"警告: 未找到538Rating.csv文件，请从FiveThirtyEight网站下载")
        print("您可以从以下网址获取: https://github.com/fivethirtyeight/data/tree/master/ncaa-forecasts")
        print("或使用Kaggle API: kaggle datasets download -d fivethirtyeight/538-ncaa-tournament-model")
        return None
    
    # 加载数据
    print(f"加载538Rating数据: {ratings_file}")
    ratings_df = pd.read_csv(ratings_file)
    
    # 基本数据清洗
    print("处理538Rating数据...")
    
    # 1. 转换列名为小写并规范化
    ratings_df.columns = [col.lower().replace(' ', '_') for col in ratings_df.columns]
    
    # 2. 查找和规范化球队ID列
    team_id_col = None
    for col in ratings_df.columns:
        if 'team' in col.lower() and 'id' in col.lower():
            team_id_col = col
            break
    
    if team_id_col is None:
        # 尝试创建与现有TeamID映射
        if 'team_name' in ratings_df.columns:
            print("未找到TeamID列，将尝试映射球队名称")
            # 此处可添加代码映射球队名称到ID
    else:
        # 确保TeamID列为整数类型
        ratings_df[team_id_col] = ratings_df[team_id_col].astype(int)
    
    # 3. 处理缺失值
    ratings_df = ratings_df.fillna({
        'offensive_rating': ratings_df['offensive_rating'].mean() if 'offensive_rating' in ratings_df.columns else 0,
        'defensive_rating': ratings_df['defensive_rating'].mean() if 'defensive_rating' in ratings_df.columns else 0,
        'overall_rating': ratings_df['overall_rating'].mean() if 'overall_rating' in ratings_df.columns else 0
    })
    
    # 4. 标准化评级值
    rating_cols = [col for col in ratings_df.columns if 'rating' in col.lower() or 'rank' in col.lower()]
    for col in rating_cols:
        if 'rank' in col.lower():  # 排名越低越好
            ratings_df[f'norm_{col}'] = (ratings_df[col] - ratings_df[col].min()) / (ratings_df[col].max() - ratings_df[col].min())
            ratings_df[f'norm_{col}'] = 1 - ratings_df[f'norm_{col}']  # 反转使得更高的值表示更好的排名
        else:  # 评级越高越好
            ratings_df[f'norm_{col}'] = (ratings_df[col] - ratings_df[col].min()) / (ratings_df[col].max() - ratings_df[col].min())
    
    print(f"538Rating数据处理完成，包含 {len(ratings_df)} 支球队的评级")
    
    return ratings_df 
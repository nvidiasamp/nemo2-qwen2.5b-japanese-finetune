#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Data Preprocessing

This module handles loading and preprocessing data for the NCAA basketball tournament prediction model.

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
Version: 2.1 ### 增加了预测所有可能的球队对阵的假设结果
Version: 3.0 ### 增加了cudf的支持
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tempfile
import hashlib
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import gpu_context, to_gpu, to_cpu
import cupy as cp
import bisect
from sklearn.model_selection import train_test_split
from data_loader import load_538_ratings


def optimize_memory_usage(df):
    """
    优化DataFrame的内存使用，根据每列的数据特性降低内存占用
    
    参数:
        df (pandas.DataFrame): 要优化的数据框
        
    返回:
        pandas.DataFrame: 优化后的数据框
    """
    # 遍历所有列
    for col in df.columns:
        # 优化整数类型
        if pd.api.types.is_integer_dtype(df[col]):
            # 确定最小可用整数类型
            col_min = df[col].min()
            col_max = df[col].max()
            
            # 根据范围选择合适的整数类型
            if col_min >= 0:
                if col_max < 2**8:
                    df[col] = df[col].astype('uint8')
                elif col_max < 2**16:
                    df[col] = df[col].astype('uint16')
                elif col_max < 2**32:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min > -2**7 and col_max < 2**7:
                    df[col] = df[col].astype('int8')
                elif col_min > -2**15 and col_max < 2**15:
                    df[col] = df[col].astype('int16')
                elif col_min > -2**31 and col_max < 2**31:
                    df[col] = df[col].astype('int32')
        
        # 优化浮点类型
        elif pd.api.types.is_float_dtype(df[col]):
            # 尝试使用float32而不是float64
            df[col] = df[col].astype('float32')
        
        # 优化对象/字符串类型
        elif pd.api.types.is_object_dtype(df[col]):
            # 如果是字符串列，检查是否可以使用类别类型
            if df[col].nunique() / len(df) < 0.5:  # 少于50%的唯一值
                df[col] = df[col].astype('category')
    
    return df


def load_data(data_path, use_cache=True, cache_dir=None):
    """
    加载NCAA篮球比赛数据集中的必要文件，并支持缓存功能
    Load necessary data files from the NCAA basketball dataset with caching support
    
    参数 Parameters:
        data_path (str): 数据文件所在的目录路径
                         Directory path containing the data files
        use_cache (bool): 是否使用缓存，默认为True
                          Whether to use cache, defaults to True
        cache_dir (str): 缓存目录，默认为None时使用系统临时目录
                         Cache directory, uses system temp directory when None
    
    返回 Returns:
        dict: 包含所有加载数据集的字典，键为数据集名称，值为对应的DataFrame
              Dictionary with all loaded datasets, keys are dataset names, values are corresponding DataFrames
    """
    # 如果未指定缓存目录，使用系统临时目录
    # Use system temp directory if cache_dir is not specified
    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    
    # 创建缓存文件路径
    # Create cache file path
    cache_hash = hashlib.md5(data_path.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"ncaa_data_cache_{cache_hash}.pkl")
    
    # 尝试从缓存加载
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        try:
            print("Loading data from cache...")
            with open(cache_file, 'rb') as f:
                data_dict = pickle.load(f)
            print("Data loaded from cache successfully.")
            for key, df in data_dict.items():
                print(f"{key}: shape {df.shape}")
            return data_dict
        except Exception as e:
            print(f"Error loading cache: {e}. Will load from original files.")
    
    print("Loading data from original files...")
    
    # 定义要加载的文件列表，使用字典映射数据集名称到文件名
    # Define files to be loaded using a dictionary mapping dataset names to filenames
    files_to_load = {
        'm_teams': 'MTeams.csv',               # 男子队伍信息 (Men's teams information)
        'w_teams': 'WTeams.csv',               # 女子队伍信息 (Women's teams information)
        'm_regular_season': 'MRegularSeasonCompactResults.csv',  # 男子常规赛结果 (Men's regular season results)
        'w_regular_season': 'WRegularSeasonCompactResults.csv',  # 女子常规赛结果 (Women's regular season results)
        'm_tourney_results': 'MNCAATourneyCompactResults.csv',   # 男子锦标赛结果 (Men's tournament results)
        'w_tourney_results': 'WNCAATourneyCompactResults.csv',   # 女子锦标赛结果 (Women's tournament results)
        'm_regular_detail': 'MRegularSeasonDetailedResults.csv', # 男子常规赛详细统计 (Men's regular season detailed stats)
        'w_regular_detail': 'WRegularSeasonDetailedResults.csv', # 女子常规赛详细统计 (Women's regular season detailed stats)
        'm_tourney_seeds': 'MNCAATourneySeeds.csv',              # 男子锦标赛种子信息 (Men's tournament seeds)
        'w_tourney_seeds': 'WNCAATourneySeeds.csv',              # 女子锦标赛种子信息 (Women's tournament seeds)
        'sample_sub': 'SampleSubmissionStage2.csv'               # 样本提交文件 (Sample submission file)
    }
    
    # 验证文件存在性，避免后续错误
    # Verify file existence to avoid subsequent errors
    for filename in files_to_load.values():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
    
    # 使用字典推导式高效地加载所有文件，避免重复代码
    # Use dictionary comprehension to efficiently load all files, avoiding code duplication
    data_dict = {key: pd.read_csv(os.path.join(data_path, filename)) 
                for key, filename in files_to_load.items() 
                if os.path.exists(os.path.join(data_path, filename))}
    
    # 在返回数据之前执行内存优化
    for key in data_dict:
        data_dict[key] = optimize_memory_usage(data_dict[key])
        print(f"优化后 {key}: 内存使用 {data_dict[key].memory_usage().sum() / 1024**2:.2f} MB")
    
    # 添加538Rating数据加载
    try:
        ratings_df = load_538_ratings(data_path, use_cache, cache_dir)
        if ratings_df is not None:
            data_dict['538_ratings'] = ratings_df
            print(f"538_ratings: shape {ratings_df.shape}")
    except Exception as e:
        print(f"加载538Rating数据时出错: {e}")
    
    # 保存到缓存
    # Save to cache
    if use_cache:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Data cached to {cache_file}")
        except Exception as e:
            print(f"Error caching data: {e}")
    
    print("Data loading complete. Dataset overview:")
    for key, df in data_dict.items():
        print(f"{key}: shape {df.shape}")
    
    return data_dict


def explore_data(data_dict, show_plots=True):
    """
    探索数据结构和基本统计特征，生成可视化图表
    Explore data structure and basic statistics, generate visualizations
    
    参数 Parameters:
        data_dict (dict): 包含所有数据集的字典
                          Dictionary containing all datasets
        show_plots (bool): 是否显示图表，默认为True
                           Whether to display plots, defaults to True
    
    返回 Returns:
        pandas.Series: 男子锦标赛种子数值，用于后续分析
                       Men's tournament seed numbers for further analysis
    """
    
    # 打印每个数据集的基本信息
    # Print basic information for each dataset
    print("Dataset overview:")
    for name, df in data_dict.items():
        print(f"\n{name} dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:")
        print(df.head(2))
    
    # 探索男子锦标赛结果按赛季分布
    # Explore men's tournament results distribution by season
    m_tourney = data_dict['m_tourney_results']
    print("\nMen's tournament results by season:")
    print(m_tourney['Season'].value_counts().sort_index())
    
    # 探索种子分布情况
    # Explore seed distribution
    m_seeds = data_dict['m_tourney_seeds']
    print("\nMen's tournament seed distribution:")
    # 提取种子数值 - 使用更高效的方法
    # Extract numeric seed values - using more efficient methods
    # 直接使用数值提取，避免正则表达式的开销
    # Directly extract numbers, avoiding regular expression overhead
    try:
        m_seeds['Seed_Number'] = m_seeds['Seed'].str[1:3].astype(int)
    except (ValueError, TypeError):
        # 如果简单转换失败，使用更健壮的方法提取数字
        import re
        def extract_seed_num(seed_str):
            if isinstance(seed_str, str):
                # 尝试提取数字部分
                nums = re.findall(r'\d+', seed_str)
                if nums:
                    return int(nums[0])
            return 8  # 默认返回中间种子值
            
        m_seeds['Seed_Number'] = m_seeds['Seed'].apply(extract_seed_num)
        
    print(m_seeds['Seed_Number'].value_counts().sort_index())
    
    # 只有在需要时才创建可视化图表，避免不必要的计算
    # Only create visualizations when needed, avoiding unnecessary computation
    if show_plots:
        plt.figure(figsize=(12, 5))
        
        # 绘制比分差异分布图 - 使用向量化操作提高性能
        # Plot score difference distribution - use vectorized operations for better performance
        plt.subplot(1, 2, 1)
        m_tourney = data_dict['m_tourney_results']
        score_diff = m_tourney['WScore'] - m_tourney['LScore']
        # 替换无穷值为NaN，使用pandas内置方法
        # Replace infinite values with NaN, using pandas built-in method
        score_diff = score_diff.replace([np.inf, -np.inf], np.nan)
        sns.histplot(score_diff, kde=True)
        plt.title('Distribution of score differences in championships (MEN)')
        plt.xlabel('Score difference')
        
        # 绘制种子与胜率关系图 - 使用更高效的数据处理方法
        # Plot seed vs win rate relationship - use more efficient data processing methods
        plt.subplot(1, 2, 2)
        
        # 使用向量化操作代替循环
        # Use vectorized operations instead of loops
        m_tourney = data_dict['m_tourney_results']
        m_seeds = data_dict['m_tourney_seeds']
        
        # 创建联合键以便进行高效连接
        # Create join keys for efficient joining
        m_seeds['key'] = m_seeds['Season'].astype(str) + '_' + m_seeds['TeamID'].astype(str)
        m_tourney['w_key'] = m_tourney['Season'].astype(str) + '_' + m_tourney['WTeamID'].astype(str)
        m_tourney['l_key'] = m_tourney['Season'].astype(str) + '_' + m_tourney['LTeamID'].astype(str)
        
        # 使用映射函数代替循环，提高性能
        # Use mapping function instead of loops for better performance
        try:
            m_seeds_dict = dict(zip(m_seeds['key'], m_seeds['Seed'].str[1:3].astype(int)))
        except (ValueError, TypeError):
            # 如果简单转换失败，使用更健壮的方法提取数字
            import re
            def extract_seed_num(seed_str):
                if isinstance(seed_str, str):
                    # 尝试提取数字部分
                    nums = re.findall(r'\d+', seed_str)
                    if nums:
                        return int(nums[0])
                return 8  # 默认返回中间种子值
                
            m_seeds_dict = dict(zip(m_seeds['key'], m_seeds['Seed'].apply(extract_seed_num)))
        
        # 映射种子值到比赛结果
        # Map seed values to game results
        m_tourney['w_seed'] = m_tourney['w_key'].map(m_seeds_dict)
        m_tourney['l_seed'] = m_tourney['l_key'].map(m_seeds_dict)
        
        # 丢弃种子值缺失的行
        # Drop rows with missing seed values
        seed_games = m_tourney.dropna(subset=['w_seed', 'l_seed'])
        
        # 向量化操作计算最小种子和低种子是否获胜
        # Vectorized operations to calculate min seed and whether lower seed won
        seed_games['min_seed'] = seed_games[['w_seed', 'l_seed']].min(axis=1)
        seed_games['lower_seed_won'] = (seed_games['w_seed'] < seed_games['l_seed']).astype(int)
        
        # 计算按最小种子分组的低种子胜率
        # Calculate lower seed win rate by min seed group
        seed_win_rates = seed_games.groupby('min_seed')['lower_seed_won'].mean()
        
        # 绘制种子与胜率关系的条形图
        # Plot bar chart of seed vs. win rate relationship
        sns.barplot(x=seed_win_rates.index, y=seed_win_rates.values)
        plt.title('The team with the lower seeds has a higher winning percentage')
        plt.xlabel('Minimum seed')
        plt.ylabel('Winning percentage')
        
        plt.tight_layout()
        plt.show()
    
    return m_seeds['Seed_Number']  # 返回种子数值用于后续分析 (Return seed numbers for later use)


def estimate_round_number(day_num):
    """
    根据比赛日期估计锦标赛轮次
    Estimate tournament round based on game date (DayNum)
    """
    # 定义轮次边界
    boundaries = [0, 136, 138, 143, 145, 150, float('inf')]
    rounds = [1, 2, 3, 4, 5, 6]
    
    # 使用二分查找确定日期所在的轮次区间
    idx = bisect.bisect_right(boundaries, day_num) - 1
    return rounds[idx]


def create_tourney_train_data(tourney_results, start_year, end_year):
    """
    从锦标赛结果创建训练数据集，使用向量化操作提高性能
    Create training dataset from tournament results, using vectorized operations for better performance
    
    参数 Parameters:
        tourney_results (pd.DataFrame): 锦标赛结果数据
                                        Tournament results data
        start_year (int): 起始年份
                          Start year for filtering
        end_year (int): 结束年份
                        End year for filtering
    
    返回 Returns:
        pd.DataFrame: 处理后的训练数据
                      Processed training data
    """
    # 使用query方法高效过滤数据
    filtered_results = tourney_results.query(f"Season >= {start_year} and Season <= {end_year}").copy()
    
    # 创建临时列用于向量化操作
    filtered_results.loc[:, 'Team1'] = filtered_results[['WTeamID', 'LTeamID']].min(axis=1)
    filtered_results.loc[:, 'Team2'] = filtered_results[['WTeamID', 'LTeamID']].max(axis=1)
    filtered_results.loc[:, 'Team1_Win'] = (filtered_results['WTeamID'] == filtered_results['Team1']).astype(int)
    
    # 向量化估计比赛轮次
    filtered_results['Round'] = filtered_results['DayNum'].apply(estimate_round_number)
    
    # 选择必要的列返回
    tourney_train_data = filtered_results[['Season', 'Team1', 'Team2', 'Team1_Win', 'Round', 'DayNum']]
    
    return tourney_train_data


def prepare_train_val_data_time_aware(X, y, tourney_train, test_size=0.2, 
                                     random_state=42, use_gpu=False):
    """准备训练/验证数据，改进数据类型处理"""
    # 确保数据类型正确
    X_np = np.array(X) if not isinstance(X, np.ndarray) else X
    y_np = np.array(y) if not isinstance(y, np.ndarray) else y
    
    # 确保y是二值的
    unique_values = np.unique(y_np)
    if len(unique_values) > 2 or not np.all(np.isin(unique_values, [0, 1])):
        print("警告: y包含非二值数据，转换为二值")
        y_np = (y_np > 0.5).astype(int)  # 将大于0.5的值转为1，其余为0
    
    # 使用分层分割
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, 
            test_size=test_size, 
            random_state=random_state,
            stratify=tourney_train['Season'].values
        )
    except Exception as e:
        print(f"分层分割失败: {e}，使用随机分割")
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, 
            test_size=test_size, 
            random_state=random_state
        )
    
    return X_train, X_val, y_train, y_val


def parallel_feature_extraction(data_dict, n_jobs=-1, verbose=1):
    """
    使用并行处理从数据中提取特征，适用于大型数据集
    Extract features from data using parallel processing, suitable for large datasets
    
    参数 Parameters:
        data_dict (dict): 包含所有数据集的字典
                          Dictionary containing all datasets
        n_jobs (int): 并行使用的CPU数量，-1表示所有CPU
                      Number of CPUs to use, -1 means all CPUs
        verbose (int): 详细程度，0=静默，1=进度条，>1=更多信息
                      Verbosity level, 0=silent, 1=progress bar, >1=more info
    
    返回 Returns:
        pd.DataFrame: 提取的特征
                      Extracted features
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    # 定义要并行处理的数据
    # Define data to be processed in parallel
    regular_season = data_dict['m_regular_season']
    teams = data_dict['m_teams']
    
    # 按团队分组以便并行处理
    # Group by team for parallel processing
    team_ids = teams['TeamID'].unique()
    
    def process_team(team_id):
        """处理单个团队的数据 (Process data for a single team)"""
        # 获取团队比赛 (Get team games)
        team_games = regular_season[(regular_season['WTeamID'] == team_id) | 
                                   (regular_season['LTeamID'] == team_id)]
        
        # 计算获胜率 (Calculate win rate)
        wins = team_games[team_games['WTeamID'] == team_id].shape[0]
        total_games = team_games.shape[0]
        win_rate = wins / total_games if total_games > 0 else 0
        
        # 计算平均得分 (Calculate average score)
        w_scores = team_games[team_games['WTeamID'] == team_id]['WScore'].sum()
        l_scores = team_games[team_games['LTeamID'] == team_id]['LScore'].sum()
        total_points = w_scores + l_scores
        avg_points = total_points / total_games if total_games > 0 else 0
        
        return {
            'TeamID': team_id,
            'WinRate': win_rate,
            'AvgPoints': avg_points,
            'TotalGames': total_games
        }
    
    # 使用并行处理提取特征
    # Extract features using parallel processing
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_team)(team_id) for team_id in tqdm(team_ids, desc="处理团队 (Processing teams)")
    )
    
    # 将结果转换为DataFrame
    # Convert results to DataFrame
    team_features = pd.DataFrame(results)
    
    return team_features


def prepare_data_with_swap(df):
    """
    准备数据并创建胜者和败者互换的副本以增加训练数据量
    
    参数:
        df (pandas.DataFrame): 原始比赛结果数据
        
    返回:
        pandas.DataFrame: 包含原始数据和互换数据的合并数据框
    """
    # 复制胜者和败者互换的数据框
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']].copy()
    if 'WFGM' in df.columns:
        # 如果存在详细统计数据，也复制这些列
        detail_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
                      'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 
                      'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        for col in detail_cols:
            if col in df.columns:
                dfswap[col] = df[col]
    
    # 交换主场/客场标记
    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    
    # 重命名列以保持一致性
    df = df.rename(columns={col: col.replace('W', 'T1_').replace('L', 'T2_') for col in df.columns})
    dfswap = dfswap.rename(columns={col: col.replace('L', 'T1_').replace('W', 'T2_') for col in dfswap.columns})
    
    # 将主场/客场标记转换为数值
    for data in [df, dfswap]:
        data.loc[data['T1_Loc'] == 'N', 'T1_Loc'] = '0'
        data.loc[data['T1_Loc'] == 'H', 'T1_Loc'] = '1'
        data.loc[data['T1_Loc'] == 'A', 'T1_Loc'] = '-1'
        if 'T1_Loc' in data.columns:
            data['T1_Loc'] = data['T1_Loc'].astype(int)
    
    # 计算得分差异
    df['PointDiff'] = df['T1_Score'] - df['T2_Score']
    dfswap['PointDiff'] = dfswap['T1_Score'] - dfswap['T2_Score']
    
    # 合并数据
    output = pd.concat([df, dfswap], ignore_index=True)
    
    return output
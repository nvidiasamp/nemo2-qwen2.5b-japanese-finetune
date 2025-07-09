#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Feature Engineering

This module handles creating and transforming features for the NCAA basketball tournament prediction model.

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
"""
import pandas as pd
import numpy as np
from scipy.special import expit
import pandas as pd
import numpy as np
from data_preprocessing import estimate_round_number
import statsmodels.api as sm


def process_seeds(tourney_seeds, start_year, end_year):
    """
    Process seed information into numerical features
    将种子信息转换为数值特征
    """
    # 预先过滤数据，避免在循环中重复筛选
    # Pre-filter data to avoid repeated filtering in the loop
    filtered_seeds = tourney_seeds[(tourney_seeds['Season'] >= start_year) & 
                                  (tourney_seeds['Season'] <= end_year)]
    
    # 定义安全的种子数字提取函数
    def safe_seed_num(seed_str):
        try:
            # 首先尝试直接提取数字部分
            if isinstance(seed_str, str) and len(seed_str) > 1:
                # 通常格式是区域字母(W,X,Y,Z)后跟数字，如W01
                return int(seed_str[1:3])
            return int(seed_str)  # 尝试直接转换
        except (ValueError, TypeError, IndexError):
            # 如果失败，尝试提取任何数字
            if isinstance(seed_str, str):
                import re
                nums = re.findall(r'\d+', seed_str)
                if nums:
                    return int(nums[0])
            # 默认返回中间种子
            return 8
    
    # 使用字典推导式替代循环创建字典
    # Use dictionary comprehension instead of loop to create dictionary
    seed_features = {
        season: {
            row['TeamID']: {
                'seed_num': safe_seed_num(row['Seed']),
                'region': row['Seed'][0] if isinstance(row['Seed'], str) and len(row['Seed']) > 0 else '',
                'seed_str': row['Seed']
            }
            for _, row in filtered_seeds[filtered_seeds['Season'] == season].iterrows()
        }
        for season in range(start_year, end_year + 1)
    }
    
    return seed_features


def calculate_team_stats(regular_season, regular_detail, start_year, end_year):
    """
    Calculate performance statistics for each team in each season
    计算每个赛季中每支队伍的表现统计数据
    """
    team_stats = {}
    
    # 预先过滤赛季数据
    regular_season_filtered = regular_season.query(f"Season >= {start_year} and Season <= {end_year}")
    
    # 处理详细数据
    if regular_detail is not None:
        regular_detail_filtered = regular_detail.query(f"Season >= {start_year} and Season <= {end_year}")
    else:
        regular_detail_filtered = None
    
    # 使用每个赛季的唯一队伍ID进行预处理
    all_season_team_ids = set()
    for season in range(start_year, end_year + 1):
        season_data = regular_season_filtered[regular_season_filtered['Season'] == season]
        season_team_ids = set(season_data['WTeamID'].unique()) | set(season_data['LTeamID'].unique())
        all_season_team_ids.update(season_team_ids)
    
    # 预先分配内存
    columns = ['num_wins', 'total_games', 'points_scored', 'points_allowed', 
              'num_losses', 'win_rate', 'avg_points_scored', 'avg_points_allowed', 
              'point_diff', 'home_wins', 'home_games', 'home_win_rate',
              'away_wins', 'away_games', 'away_win_rate', 'recent_win_rate', 
              'momentum', 'season_rank', 'normalized_rank']
    
    for season in range(start_year, end_year + 1):
        season_data = regular_season_filtered[regular_season_filtered['Season'] == season]
        
        # 使用向量化操作创建胜负数据框
        # Use vectorized operations to create win/loss dataframes
        wins_df = season_data[['WTeamID', 'WScore', 'LScore', 'WLoc', 'DayNum']].rename(
            columns={'WTeamID': 'TeamID', 'WScore': 'PointsScored', 'LScore': 'PointsAllowed', 'WLoc': 'Loc'}
        ).assign(Win=1)
        
        losses_df = season_data[['LTeamID', 'LScore', 'WScore', 'WLoc', 'DayNum']].rename(
            columns={'LTeamID': 'TeamID', 'LScore': 'PointsScored', 'WScore': 'PointsAllowed'}
        ).assign(Win=0)
        
        # 转换主场/客场标记
        # Transform home/away indicators
        losses_df['Loc'] = losses_df['WLoc'].map({'H': 'A', 'A': 'H', 'N': 'N'})
        
        # 合并所有比赛数据
        # Merge all games data
        all_games = pd.concat([wins_df, losses_df]).reset_index(drop=True)
        
        # 使用groupby进行聚合计算
        # Use groupby for aggregation calculations
        basic_stats = all_games.groupby('TeamID').agg(
            num_wins=('Win', 'sum'),
            total_games=('Win', 'count'),
            points_scored=('PointsScored', 'sum'),
            points_allowed=('PointsAllowed', 'sum')
        )
        
        # 计算派生统计数据
        # Calculate derived statistics
        basic_stats['num_losses'] = basic_stats['total_games'] - basic_stats['num_wins']
        basic_stats['win_rate'] = basic_stats['num_wins'] / basic_stats['total_games']
        basic_stats['avg_points_scored'] = basic_stats['points_scored'] / basic_stats['total_games']
        basic_stats['avg_points_allowed'] = basic_stats['points_allowed'] / basic_stats['total_games']
        basic_stats['point_diff'] = basic_stats['avg_points_scored'] - basic_stats['avg_points_allowed']
        
        # 主场/客场统计数据
        # Home/away statistics
        home_stats = all_games[all_games['Loc'] == 'H'].groupby('TeamID').agg(
            home_wins=('Win', 'sum'),
            home_games=('Win', 'count')
        )
        home_stats['home_win_rate'] = home_stats['home_wins'] / home_stats['home_games']
        
        away_stats = all_games[all_games['Loc'] == 'A'].groupby('TeamID').agg(
            away_wins=('Win', 'sum'),
            away_games=('Win', 'count')
        )
        away_stats['away_win_rate'] = away_stats['away_wins'] / away_stats['away_games']
        
        # 合并基础统计数据
        # Merge basic statistics
        team_season_stats = pd.merge(basic_stats, home_stats, on='TeamID', how='left')
        team_season_stats = pd.merge(team_season_stats, away_stats, on='TeamID', how='left')
        
        # 处理NaN值（对于没有主场或客场比赛的队伍）
        # Handle NaN values (for teams without home or away games)
        team_season_stats = team_season_stats.fillna({
            'home_win_rate': 0, 
            'away_win_rate': 0,
            'home_wins': 0,
            'home_games': 0,
            'away_wins': 0,
            'away_games': 0
        })
        
        # 转换为字典前减少内存使用
        # 使用浮点精度降低
        float_cols = ['win_rate', 'avg_points_scored', 'avg_points_allowed', 
                     'point_diff', 'home_win_rate', 'away_win_rate', 
                     'recent_win_rate', 'momentum', 'normalized_rank']
        
        for col in float_cols:
            if col in team_season_stats:
                team_season_stats[col] = team_season_stats[col].astype('float32')
        
        # 使用最优的整数类型
        int_cols = ['num_wins', 'total_games', 'home_wins', 'home_games', 
                   'away_wins', 'away_games', 'season_rank']
        
        for col in int_cols:
            if col in team_season_stats:
                team_season_stats[col] = team_season_stats[col].astype('int32')
                
        # 转换为字典
        teams_season_stats = team_season_stats.to_dict('index')
        team_stats[season] = teams_season_stats
    
    return team_stats


def create_matchup_history(regular_season, tourney_results, start_year, end_year):
    """
    Create features based on historical matchups
    基于历史对战创建特征
    """
    # 合并常规赛和锦标赛结果
    # Merge regular season and tournament results
    common_columns = list(set(regular_season.columns) & set(tourney_results.columns))
    regular_season_filtered = regular_season[common_columns].dropna(axis=1, how='all')
    tourney_results_filtered = tourney_results[common_columns].dropna(axis=1, how='all')
    
    all_games = pd.concat(
        [regular_season_filtered, tourney_results_filtered], 
        axis=0,
        join='inner',
        ignore_index=True
    )
    
    all_games = all_games[(all_games['Season'] >= start_year) & 
                          (all_games['Season'] <= end_year)]
    
    # 初始化结果字典
    # Initialize result dictionary
    matchup_history = {season: {} for season in range(start_year, end_year + 1)}
    
    # 为每个对战创建唯一键
    # Create unique key for each matchup
    all_games['Team1'] = all_games[['WTeamID', 'LTeamID']].min(axis=1)
    all_games['Team2'] = all_games[['WTeamID', 'LTeamID']].max(axis=1)
    all_games['Team1Won'] = (all_games['WTeamID'] == all_games['Team1']).astype(int)
    
    # 使用groupby聚合计算对战历史
    # Use groupby aggregation to calculate matchup history
    grouped = all_games.groupby(['Season', 'Team1', 'Team2'])
    
    for (season, team1, team2), group in grouped:
        # 计算对战统计
        # Calculate matchup statistics
        games_count = len(group)
        team1_wins = group['Team1Won'].sum()
        
        team1_points = group.apply(
            lambda x: x['WScore'] if x['WTeamID'] == team1 else x['LScore'], 
            axis=1
        ).sum()
        
        team2_points = group.apply(
            lambda x: x['WScore'] if x['WTeamID'] == team2 else x['LScore'], 
            axis=1
        ).sum()
        
        # 存储为字典
        # Store as dictionary
        matchup_key = (team1, team2)
        matchup_history[season][matchup_key] = {
            'games': games_count,
            'wins_team1': team1_wins,
            'points_team1': team1_points,
            'points_team2': team2_points,
            'avg_point_diff': (team1_points - team2_points) / games_count
        }
    
    return matchup_history


def calculate_progression_probabilities(seed_features, team_stats):
    """
    Calculate progression probabilities for each team in various rounds
    计算各轮次中每支队伍的晋级概率
    """
    # 基于历史数据的种子晋级率（近似值）
    # Seed progression rates based on historical data (approximate values)
    seed_progression_rates = {
        1: [0.99, 0.93, 0.74, 0.52, 0.36, 0.22],  # 1号种子在每轮中的晋级率
        2: [0.96, 0.82, 0.56, 0.36, 0.23, 0.12],
        3: [0.94, 0.72, 0.44, 0.22, 0.12, 0.06],
        4: [0.92, 0.64, 0.34, 0.18, 0.08, 0.04],
        5: [0.85, 0.47, 0.24, 0.12, 0.04, 0.02],
        6: [0.76, 0.36, 0.18, 0.08, 0.03, 0.01],
        7: [0.72, 0.32, 0.15, 0.06, 0.02, 0.01],
        8: [0.56, 0.22, 0.10, 0.04, 0.01, 0.005],
        9: [0.44, 0.18, 0.08, 0.03, 0.01, 0.004],
        10: [0.68, 0.28, 0.12, 0.05, 0.02, 0.007],
        11: [0.59, 0.24, 0.10, 0.04, 0.01, 0.005],
        12: [0.36, 0.15, 0.06, 0.02, 0.007, 0.002],
        13: [0.20, 0.08, 0.03, 0.01, 0.003, 0.001],
        14: [0.16, 0.05, 0.02, 0.005, 0.001, 0.0005],
        15: [0.06, 0.02, 0.005, 0.001, 0.0003, 0.0001],
        16: [0.01, 0.003, 0.0005, 0.0001, 0.00002, 0.000005],
    }
    
    # 创建字典推导式初始化结果
    # Create dictionary comprehension to initialize results
    progression_probs = {}
    
    # 使用向量化操作计算晋级概率
    # Use vectorized operations to calculate progression probabilities
    for season in seed_features:
        season_probs = {}
        
        for team_id, seed_info in seed_features[season].items():
            seed_num = seed_info['seed_num']
            
            # 查找种子号对应的基础晋级率
            # Look up base progression rates for seed number
            base_rates = seed_progression_rates.get(
                seed_num, 
                seed_progression_rates[min(seed_progression_rates.keys(), key=lambda k: abs(k-seed_num))]
            )
            
            # 应用队伍实力调整
            # Apply team strength adjustments
            team_strength = team_stats.get(season, {}).get(team_id, {})
            
            # 计算实力调整因子
            # Calculate strength adjustment factor
            strength_factor = 1.0
            
            if team_strength:
                # 根据胜率调整
                # Adjust based on win rate
                win_rate_adj = (team_strength.get('win_rate', 0.5) - 0.5) * 0.3
                strength_factor += win_rate_adj
                
                # 根据得分差异调整
                # Adjust based on point difference
                point_diff_adj = team_strength.get('point_diff', 0) * 0.02
                strength_factor += point_diff_adj
                
                # 根据排名调整
                # Adjust based on ranking
                rank_adj = (1 - team_strength.get('normalized_rank', 0.5)) * 0.15
                strength_factor += rank_adj
            
            # 应用调整，但保持在合理范围内
            # Apply adjustment, but keep within reasonable range
            adjusted_rates = [min(0.999, max(0.001, rate * strength_factor)) for rate in base_rates]
            
            # 存储晋级概率
            # Store progression probabilities
            team_progression = {f'rd{i+1}_win': rate for i, rate in enumerate(adjusted_rates)}
            season_probs[team_id] = team_progression
        
        progression_probs[season] = season_probs
    
    return progression_probs


def convert_progression_to_matchup(team1_prog, team2_prog, round_num):
    """
    Convert progression probabilities to matchup probabilities
    将晋级概率转换为对战概率（实现goto_conversion概念）
    
    Parameters:
    -----------
    team1_prog : dict
        Team 1 progression probabilities
        队伍1的晋级概率
    team2_prog : dict
        Team 2 progression probabilities
        队伍2的晋级概率
    round_num : int
        Tournament round number
        锦标赛轮次编号
        
    Returns:
    --------
    float
        Probability of team1 winning against team2
        队伍1战胜队伍2的概率
    """
    # 获取当前轮次和前一轮次的键
    # Get current round and previous round keys
    curr_round_key = f'rd{round_num}_win'
    prev_round_key = f'rd{round_num-1}_win' if round_num > 1 else None
    
    # 获取条件晋级概率
    # Get conditional progression probabilities
    if round_num == 1 or prev_round_key is None:
        # 第一轮直接使用基础概率
        # For first round, use base probabilities directly
        team1_win_given_reach = team1_prog.get(curr_round_key, 0.5)
        team2_win_given_reach = team2_prog.get(curr_round_key, 0.5)
    else:
        # 计算条件概率：P(晋级到下一轮|已经到达当前轮)
        # Calculate conditional probability: P(advance to next round | already reached current round)
        prev_t1 = team1_prog.get(prev_round_key, 0.001)
        prev_t2 = team2_prog.get(prev_round_key, 0.001)
        
        curr_t1 = team1_prog.get(curr_round_key, 0.0005)
        curr_t2 = team2_prog.get(curr_round_key, 0.0005)
        
        # 安全地计算条件概率
        # Safely calculate conditional probabilities
        team1_win_given_reach = curr_t1 / prev_t1 if prev_t1 > 0 else 0.5
        team2_win_given_reach = curr_t2 / prev_t2 if prev_t2 > 0 else 0.5
    
    # 确保概率在有效范围内
    # Ensure probabilities are in valid range
    team1_win_given_reach = max(0.001, min(0.999, team1_win_given_reach))
    team2_win_given_reach = max(0.001, min(0.999, team2_win_given_reach))
    
    # 将条件概率转换为对战概率
    # Convert conditional probabilities to matchup probabilities
    raw_sum = team1_win_given_reach + team2_win_given_reach
    
    # 如果概率总和接近1，不需要太多调整
    # If probability sum is close to 1, not much adjustment needed
    if 0.95 <= raw_sum <= 1.05:
        # 简单归一化
        # Simple normalization
        team1_matchup_prob = team1_win_given_reach / raw_sum
    else:
        # 应用偏差校正（热门队伍被低估，冷门队伍被高估）
        # Apply bias correction (favorite teams underestimated, longshot teams overestimated)
        if team1_win_given_reach > team2_win_given_reach:
            # 队伍1是热门，给予额外权重
            # team1 is favorite, give extra weight
            ratio = team1_win_given_reach / team2_win_given_reach
            boosted_ratio = ratio ** 1.1  # 为热门队伍增加权重 (Boost for favorite teams)
            team1_matchup_prob = boosted_ratio / (1 + boosted_ratio)
        else:
            # 队伍2是热门
            # team2 is favorite
            ratio = team2_win_given_reach / team1_win_given_reach
            boosted_ratio = ratio ** 1.1  # 为热门队伍增加权重 (Boost for favorite teams)
            team1_matchup_prob = 1 / (1 + boosted_ratio)
    
    return team1_matchup_prob


def build_matchup_features(team1, team2, features_dict, target_year=2025, round_num=2, gender='men', data_dict=None):
    """
    为两支队伍的对阵创建特征，优化类型处理与性能
    """
    # 错误处理：确保输入有效
    if team1 is None or team2 is None:
        print(f"警告：队伍ID无效 - team1: {team1}, team2: {team2}")
        return None
    
    try:
        # 确保团队ID为整数
        team1 = int(team1)
        team2 = int(team2)
    except (ValueError, TypeError) as e:
        print(f"错误：队伍ID转换失败 - {e}")
        return None
    
    # 确保team1 < team2
    if team1 > team2:
        team1, team2 = team2, team1  # 交换以保持一致性
    
    # 提取队伍统计数据，避免重复访问字典
    team_stats = features_dict.get('team_stats', {}).get(target_year, {})
    seed_features = features_dict.get('seed_features', {}).get(target_year, {})
    matchup_history = features_dict.get('matchup_history', {}).get(target_year, {})
    progression_probs = features_dict.get('progression_probs', {}).get(target_year, {})
    
    # 如果特征数据缺失，使用默认值
    if not team_stats or team1 not in team_stats or team2 not in team_stats:
        # 创建默认特征
        return {
            # 基本特征
            'Team1': team1,
            'Team2': team2,
            'Round': round_num if round_num is not None else 2,  # 默认轮次
            'Season': target_year,
            # 统计特征（使用中位数估计）
            'WinRate_1': 0.5,
            'WinRate_2': 0.5,
            'AvgScore_1': 70.0,
            'AvgScore_2': 70.0,
            'AvgScoreAllowed_1': 70.0,
            'AvgScoreAllowed_2': 70.0,
            'ScoreDiff_1': 0.0,
            'ScoreDiff_2': 0.0,
            'WinRateDiff': 0.0,
            'ScoreDiffDiff': 0.0,
            # 种子特征
            'Seed_1': 8,  # 使用中间种子作为默认值
            'Seed_2': 8,
            'SeedDiff': 0,
            # 进度特征
            'ProgProb_1': 0.5,
            'ProgProb_2': 0.5,
            'ProgProbDiff': 0.0,
        }
    
    # 提前获取所有需要的数据，减少字典查找
    t1_stats = team_stats.get(team1, {})
    t2_stats = team_stats.get(team2, {})
    
    # 使用单次字典查找创建所有特征，减少访问开销
    win_rate_1 = t1_stats.get('win_rate', 0.5)
    win_rate_2 = t2_stats.get('win_rate', 0.5)
    avg_pts_1 = t1_stats.get('avg_points_scored', 70.0)
    avg_pts_2 = t2_stats.get('avg_points_scored', 70.0)
    avg_pts_allowed_1 = t1_stats.get('avg_points_allowed', 70.0)
    avg_pts_allowed_2 = t2_stats.get('avg_points_allowed', 70.0)
    score_diff_1 = t1_stats.get('point_diff', 0.0)
    score_diff_2 = t2_stats.get('point_diff', 0.0)
    
    # 获取种子信息，使用默认值如果不存在
    seed_1 = seed_features.get(team1, {}).get('seed_num', 8)
    seed_2 = seed_features.get(team2, {}).get('seed_num', 8)
    
    # 确保种子是整数
    try:
        seed_1 = int(seed_1) if seed_1 is not None else 8
        seed_2 = int(seed_2) if seed_2 is not None else 8
    except (ValueError, TypeError):
        # 如果是字符串但包含数字，尝试提取数字部分
        if isinstance(seed_1, str):
            import re
            nums = re.findall(r'\d+', seed_1)
            seed_1 = int(nums[0]) if nums else 8
        else:
            seed_1 = 8
            
        if isinstance(seed_2, str):
            import re
            nums = re.findall(r'\d+', seed_2)
            seed_2 = int(nums[0]) if nums else 8
        else:
            seed_2 = 8
    
    # 获取进度概率
    prog_1 = progression_probs.get(team1, {}).get(f'rd{round_num}_win', 0.5)
    prog_2 = progression_probs.get(team2, {}).get(f'rd{round_num}_win', 0.5)
    
    # 历史对阵信息
    matchup_key = (team1, team2)
    history = matchup_history.get(matchup_key, {})
    games = history.get('games', 0)
    team1_wins = history.get('wins_team1', 0)
    
    # 一次性创建并返回特征字典，减少操作
    return {
        # 基本特征
        'Team1': team1,
        'Team2': team2,
        'Round': round_num if round_num is not None else 2,
        'Season': target_year,
        # 团队统计特征
        'WinRate_1': win_rate_1,
        'WinRate_2': win_rate_2,
        'WinRateDiff': win_rate_1 - win_rate_2,
        'AvgScore_1': avg_pts_1,
        'AvgScore_2': avg_pts_2,
        'AvgScoreAllowed_1': avg_pts_allowed_1,
        'AvgScoreAllowed_2': avg_pts_allowed_2,
        'ScoreDiff_1': score_diff_1,
        'ScoreDiff_2': score_diff_2,
        'ScoreDiffDiff': score_diff_1 - score_diff_2,
        # 种子特征
        'Seed_1': seed_1,
        'Seed_2': seed_2,
        'SeedDiff': seed_1 - seed_2,
        # 进度特征
        'ProgProb_1': prog_1,
        'ProgProb_2': prog_2,
        'ProgProbDiff': prog_1 - prog_2,
        # 历史对阵特征
        'PreviousGames': games,
        'Team1PrevWins': team1_wins,
        'Team1WinPct': (team1_wins / games) if games > 0 else 0.5,
        'AvgPointDiff': history.get('avg_point_diff', 0)
    }


def calculate_team_quality(regular_season, start_year, end_year, use_cache=True, cache_dir=None):
    """
    使用广义线性模型计算每个球队的质量分数
    
    参数:
        regular_season (pd.DataFrame): 常规赛结果数据
        start_year (int): 起始年份
        end_year (int): 结束年份
        use_cache (bool): 是否使用缓存
        cache_dir (str): 缓存目录
        
    返回:
        dict: 每个赛季每个队伍的质量分数
    """
    # 如果未指定缓存目录，使用系统临时目录
    if cache_dir is None and use_cache:
        import tempfile
        cache_dir = tempfile.gettempdir()
    
    # 创建缓存文件路径
    if use_cache:
        import hashlib
        import os
        cache_hash = hashlib.md5(f"{start_year}_{end_year}".encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"team_quality_cache_{cache_hash}.pkl")
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    import pickle
                    quality_dict = pickle.load(f)
                print("团队质量数据已从缓存加载")
                return quality_dict
            except Exception as e:
                print(f"加载缓存时出错: {e}。将重新计算团队质量。")
    
    # 预处理常规赛数据
    regular_season_effects = regular_season.copy()
    regular_season_effects = regular_season_effects[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']]
    regular_season_effects['PointDiff'] = regular_season_effects['WScore'] - regular_season_effects['LScore']
    regular_season_effects['win'] = 1  # 赢家始终为1
    
    # 初始化结果字典
    quality_dict = {}
    
    # 针对每个赛季计算团队质量
    for season in range(start_year, end_year + 1):
        print(f"计算{season}赛季的团队质量...")
        season_data = regular_season_effects[regular_season_effects['Season'] == season].copy()
        
        if len(season_data) < 10:
            print(f"警告: {season}赛季数据不足，跳过")
            continue
            
        # 将团队ID转换为字符串类型，以便在公式中使用
        season_data['WTeamID'] = 'T1_' + season_data['WTeamID'].astype(str)
        season_data['LTeamID'] = 'T2_' + season_data['LTeamID'].astype(str)
        
        try:
            # 使用广义线性模型估计团队质量
            formula = 'win ~ -1 + WTeamID + LTeamID'
            glm = sm.GLM.from_formula(
                formula=formula,
                data=season_data,
                family=sm.families.Binomial()
            ).fit()
            
            # 提取参数
            quality = pd.DataFrame(glm.params).reset_index()
            quality.columns = ['TeamID', 'quality']
            
            # 仅保留胜者团队参数
            quality = quality[quality['TeamID'].str.startswith('T1_')].copy()
            quality['TeamID'] = quality['TeamID'].str[3:].astype(int)  # 移除"T1_"前缀
            
            # 存储到结果字典
            quality_dict[season] = dict(zip(quality['TeamID'], quality['quality']))
            
        except Exception as e:
            print(f"计算{season}赛季团队质量时出错: {e}")
    
    # 保存到缓存
    if use_cache:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                import pickle
                pickle.dump(quality_dict, f)
            print(f"团队质量数据已缓存到 {cache_file}")
        except Exception as e:
            print(f"缓存团队质量数据时出错: {e}")
    
    return quality_dict


def integrate_538_ratings(features_dict, data_dict, start_year, end_year):
    """
    整合538评级数据到特征工程流程中
    
    参数:
        features_dict: 现有特征字典
        data_dict: 数据字典
        start_year: 开始年份
        end_year: 结束年份
        
    返回:
        更新后的特征字典
    """
    ratings_df = data_dict.get('538_ratings')
    if ratings_df is None:
        print("未找到538Rating数据，跳过该特征集成")
        return features_dict
    
    print("整合538Rating评级数据到特征中...")
    
    # 找到正确的ID列
    team_id_col = None
    for col in ratings_df.columns:
        if 'team' in col.lower() and 'id' in col.lower():
            team_id_col = col
            break
    
    if team_id_col is None:
        print("错误: 无法在538Rating数据中找到团队ID列")
        return features_dict
    
    # 创建季节性的评级映射
    team_ratings = {}
    for year in range(start_year, end_year + 1):
        # 如果数据包含season/year列，则按季节筛选
        if 'season' in ratings_df.columns:
            year_ratings = ratings_df[ratings_df['season'] == year]
        elif 'year' in ratings_df.columns:
            year_ratings = ratings_df[ratings_df['year'] == year]
        else:
            # 否则使用所有数据
            year_ratings = ratings_df
        
        # 为每个队伍创建评级字典
        year_dict = {}
        for _, row in year_ratings.iterrows():
            team_id = row[team_id_col]
            
            # 收集所有评级特征
            rating_features = {}
            for col in row.index:
                if 'rating' in col.lower() or 'rank' in col.lower() or 'norm_' in col.lower():
                    rating_features[col] = row[col]
            
            year_dict[team_id] = rating_features
        
        team_ratings[year] = year_dict
    
    # 更新特征字典
    if '538_ratings' not in features_dict:
        features_dict['538_ratings'] = {}
    
    features_dict['538_ratings'] = team_ratings
    
    print(f"538Rating特征已添加到特征字典中，覆盖 {len(team_ratings)} 个季节")
    
    return features_dict


def create_538_time_series_features(features_dict, data_dict, start_year, end_year):
    """
    创建基于时间序列的538评级特征
    
    参数:
        features_dict: 特征字典
        data_dict: 数据字典
        start_year: 开始年份
        end_year: 结束年份
        
    返回:
        更新的特征字典
    """
    if '538_ratings' not in features_dict:
        print("未找到538评级数据，无法创建时间序列特征")
        return features_dict
    
    print("创建538时间序列特征...")
    
    # 创建时间序列特征存储
    time_series = {}
    
    # 遍历每个队伍
    all_teams = set()
    for year in features_dict['538_ratings']:
        all_teams.update(features_dict['538_ratings'][year].keys())
    
    for team_id in all_teams:
        team_ts = {}
        
        # 收集该队伍在各个季节的评级
        ratings_by_year = {}
        for year in range(start_year, end_year + 1):
            if year in features_dict['538_ratings'] and team_id in features_dict['538_ratings'][year]:
                ratings_by_year[year] = features_dict['538_ratings'][year][team_id]
        
        # 如果至少有两个季节的数据，计算趋势
        if len(ratings_by_year) >= 2:
            years = sorted(ratings_by_year.keys())
            
            # 计算整体评级趋势
            if 'overall_rating' in ratings_by_year[years[0]]:
                overall_trend = []
                for i in range(1, len(years)):
                    curr = ratings_by_year[years[i]].get('overall_rating', 0)
                    prev = ratings_by_year[years[i-1]].get('overall_rating', 0)
                    overall_trend.append(curr - prev)
                
                if overall_trend:
                    team_ts['overall_rating_trend'] = sum(overall_trend) / len(overall_trend)
                    team_ts['overall_rating_volatility'] = np.std(overall_trend) if len(overall_trend) > 1 else 0
            
            # 计算进攻评级趋势
            if 'offensive_rating' in ratings_by_year[years[0]]:
                offensive_trend = []
                for i in range(1, len(years)):
                    curr = ratings_by_year[years[i]].get('offensive_rating', 0)
                    prev = ratings_by_year[years[i-1]].get('offensive_rating', 0)
                    offensive_trend.append(curr - prev)
                
                if offensive_trend:
                    team_ts['offensive_rating_trend'] = sum(offensive_trend) / len(offensive_trend)
            
            # 计算防守评级趋势
            if 'defensive_rating' in ratings_by_year[years[0]]:
                defensive_trend = []
                for i in range(1, len(years)):
                    curr = ratings_by_year[years[i]].get('defensive_rating', 0)
                    prev = ratings_by_year[years[i-1]].get('defensive_rating', 0)
                    defensive_trend.append(curr - prev)
                
                if defensive_trend:
                    team_ts['defensive_rating_trend'] = sum(defensive_trend) / len(defensive_trend)
        
        # 保存该队伍的时间序列特征
        if team_ts:
            time_series[team_id] = team_ts
    
    # 将时间序列特征添加到特征字典
    features_dict['538_time_series'] = time_series
    
    print(f"创建了 {len(time_series)} 支队伍的538时间序列特征")
    
    return features_dict
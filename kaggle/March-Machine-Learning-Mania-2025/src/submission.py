#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Prediction and Submission

This module handles generating predictions for tournament games and creating submission files.

Author: Junming Zhao
Date: 2025-03-13    
Version: 2.0 ### 增加了针对女队的预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from evaluate import apply_brier_optimal_strategy
from feature_engineering import convert_progression_to_matchup
from train_model import add_favorite_longshot_features
import os
from functools import lru_cache

# 全局变量定义
MENS_TEAMS_IDS = set()
WOMENS_TEAMS_IDS = set()

def check_sample_submission_format(sample_submission):
    """
    分析样本提交文件的格式
    
    参数:
        sample_submission (pd.DataFrame): 样本提交数据框
        
    返回:
        dict: 包含格式信息的字典
    """
    # 分析ID格式和范围
    id_format = {}
    
    # 提取ID格式示例
    id_examples = sample_submission['ID'].head(5).tolist()
    
    # 计算男女队比赛比例
    men_count = sum(1 for id in sample_submission['ID'] if '_men' in id)
    women_count = sum(1 for id in sample_submission['ID'] if '_women' in id)
    
    print(f"样本提交格式分析:")
    print(f"- 总行数: {len(sample_submission)}")
    print(f"- ID示例: {', '.join(id_examples)}")
    print(f"- 男子队比赛: {men_count}")
    print(f"- 女子队比赛: {women_count}")
    
    return {
        'total_rows': len(sample_submission),
        'id_examples': id_examples,
        'men_count': men_count,
        'women_count': women_count
    }


def build_matchup_features(team1, team2, features_dict, target_year=2025, round_num=2, gender='men', data_dict=None):
    """构建队伍对阵的特征，加入538时间序列特征"""
    # 确保team1 < team2，符合比赛要求
    swap_teams = False
    if team1 > team2:
        team1, team2 = team2, team1
        swap_teams = True
    
    # 获取最近可用赛季的特征
    latest_season = max(features_dict['team_stats'].keys()) if 'team_stats' in features_dict and features_dict['team_stats'] else target_year-1
    team_stats = features_dict.get('team_stats', {}).get(latest_season, {})
    seed_features = features_dict.get('seed_features', {}).get(latest_season, {})
    progression_probs = features_dict.get('progression_probs', {}).get(latest_season, {})
    matchup_history = features_dict.get('matchup_history', {})
    
    # 收集两支队伍的特征
    features = {}
    
    # 基本特征 - 队伍ID
    features['team1_id'] = team1
    features['team2_id'] = team2
    
    # 性别特征 - 如果提供了data_dict则使用
    match_gender = gender  # 默认使用传入的性别参数
    if data_dict is not None:
        team1_gender = identify_team_gender(team1, data_dict)
        team2_gender = identify_team_gender(team2, data_dict)
        features['team1_gender'] = 1 if team1_gender == 'men' else 0
        features['team2_gender'] = 1 if team2_gender == 'men' else 0
        
        # 检查队伍性别是否一致，如果不一致使用第一支队伍的性别
        if team1_gender != team2_gender:
            print(f"警告：队伍性别不匹配 - team1({team1}): {team1_gender}, team2({team2}): {team2_gender}，使用{team1_gender}作为匹配性别")
            match_gender = team1_gender
    
    # 队伍统计数据
    for team_id, prefix in [(team1, 'team1'), (team2, 'team2')]:
        if team_id in team_stats:
            stats = team_stats[team_id]
            # 添加所有可用统计数据
            for stat_name, stat_value in stats.items():
                features[f'{prefix}_{stat_name}'] = stat_value
        else:
            # 扩展默认值列表
            default_stats = {
                'win_rate': 0.5,
                'point_diff': 0.0,
                'num_wins': 0,
                'num_losses': 0,
                'total_games': 0,
                'home_win_rate': 0.5,
                'away_win_rate': 0.5,
                'recent_win_rate': 0.5,
                'momentum': 0.0,
                'season_rank': 0,
                'normalized_rank': 0.5
            }
            # 添加所有默认值
            for stat_name, stat_value in default_stats.items():
                features[f'{prefix}_{stat_name}'] = stat_value
    
    # 种子特征
    for team_id, prefix in [(team1, 'team1'), (team2, 'team2')]:
        if team_id in seed_features:
            seed_info = seed_features[team_id]
            features[f'{prefix}_seed'] = seed_info['seed_num']
            features[f'{prefix}_region'] = ord(seed_info['region']) - ord('A')  # 转换为数值
        else:
            # 默认值
            features[f'{prefix}_seed'] = 16  # 假设最低种子
            features[f'{prefix}_region'] = 0
    
    # 计算特征差异
    features['seed_diff'] = features.get('team1_seed', 16) - features.get('team2_seed', 16)
    features['win_rate_diff'] = features.get('team1_win_rate', 0.5) - features.get('team2_win_rate', 0.5)
    features['point_diff_diff'] = features.get('team1_point_diff', 0.0) - features.get('team2_point_diff', 0.0)
    
    # 历史对阵
    matchup_key = (team1, team2)
    for season, matchups in matchup_history.items():
        if matchup_key in matchups:
            history = matchups[matchup_key]
            features['previous_matchups'] = history['games']
            features['team1_win_rate_h2h'] = history['wins_team1'] / history['games'] if history['games'] > 0 else 0.5
            features['avg_point_diff_h2h'] = history['avg_point_diff']
            break
    else:
        # 无历史对阵
        features['previous_matchups'] = 0
        features['team1_win_rate_h2h'] = 0.5
        features['avg_point_diff_h2h'] = 0.0
    
    # 晋级概率
    if progression_probs:
        team1_prog = progression_probs.get(team1, {})
        team2_prog = progression_probs.get(team2, {})
        features['team1_rd_win_prob'] = team1_prog.get(f'rd{round_num}_win', 0.5)
        features['team2_rd_win_prob'] = team2_prog.get(f'rd{round_num}_win', 0.5)
        
        # 计算对阵概率
        features['progression_based_prob'] = convert_progression_to_matchup(
            team1_prog, team2_prog, round_num
        )
    
    # 添加538特征
    if '538_ratings' in features_dict:
        # 获取最近一个可用的评级年份
        rating_years = list(features_dict['538_ratings'].keys())
        latest_year = max([year for year in rating_years if year <= target_year]) if rating_years else None
        
        if latest_year:
            ratings_dict = features_dict['538_ratings'][latest_year]
            
            # 为两支队伍添加538特征
            for team_id, prefix in [(team1, 'team1'), (team2, 'team2')]:
                if team_id in ratings_dict:
                    team_ratings = ratings_dict[team_id]
                    
                    # 添加所有评级特征
                    for rating_name, rating_value in team_ratings.items():
                        # 使用规范化的特征名称
                        feature_name = f"{prefix}_538_{rating_name.lower().replace(' ', '_')}"
                        features[feature_name] = rating_value
                else:
                    # 如果没有找到评级，使用默认值
                    features[f"{prefix}_538_overall_rating"] = 0.5
                    features[f"{prefix}_538_offensive_rating"] = 0.5
                    features[f"{prefix}_538_defensive_rating"] = 0.5
            
            # 计算评级差异特征
            for rating_type in ['overall_rating', 'offensive_rating', 'defensive_rating']:
                team1_feat = f"team1_538_{rating_type}"
                team2_feat = f"team2_538_{rating_type}"
                
                if team1_feat in features and team2_feat in features:
                    features[f"538_{rating_type}_diff"] = features[team1_feat] - features[team2_feat]
                    
            # 添加538看好的队伍获胜概率估计
            if 'team1_538_overall_rating' in features and 'team2_538_overall_rating' in features:
                t1_rating = features['team1_538_overall_rating']
                t2_rating = features['team2_538_overall_rating']
                if t1_rating + t2_rating > 0:
                    features['538_win_prob'] = t1_rating / (t1_rating + t2_rating)
                else:
                    features['538_win_prob'] = 0.5
    
    # 添加538时间序列特征
    if '538_time_series' in features_dict:
        ts_features = features_dict['538_time_series']
        
        # 为两支队伍添加时间序列特征
        for team_id, prefix in [(team1, 'team1'), (team2, 'team2')]:
            if team_id in ts_features:
                team_ts = ts_features[team_id]
                
                # 添加所有时间序列特征
                for ts_name, ts_value in team_ts.items():
                    features[f"{prefix}_538_ts_{ts_name}"] = ts_value
            else:
                # 默认值
                features[f"{prefix}_538_ts_overall_rating_trend"] = 0
                features[f"{prefix}_538_ts_offensive_rating_trend"] = 0
                features[f"{prefix}_538_ts_defensive_rating_trend"] = 0
        
        # 计算趋势差异
        for trend_type in ['overall_rating_trend', 'offensive_rating_trend', 'defensive_rating_trend']:
            team1_feat = f"team1_538_ts_{trend_type}"
            team2_feat = f"team2_538_ts_{trend_type}"
            
            if team1_feat in features and team2_feat in features:
                features[f"538_ts_{trend_type}_diff"] = features[team1_feat] - features[team2_feat]
    
    # 如果交换了队伍，不需要额外处理
    # 因为我们确保了team1 < team2，这对应于要求的预测格式
    
    return features


# 添加新函数用于识别队伍ID性别
def identify_team_gender(team_id, data_dict=None):
    """增强的队伍性别识别函数"""
    if data_dict is None:
        return 'men'  # 默认为男队
    
    # 优先使用缓存的全局集合
    global MENS_TEAMS_IDS, WOMENS_TEAMS_IDS
    if MENS_TEAMS_IDS and WOMENS_TEAMS_IDS:
        if team_id in MENS_TEAMS_IDS:
            return 'men'
        elif team_id in WOMENS_TEAMS_IDS:
            return 'women'
    
    # 回退到检查数据字典
    m_teams = data_dict.get('m_teams', None)
    w_teams = data_dict.get('w_teams', None)
    
    # 使用整数或字符串类型都进行检查
    team_id_int = team_id
    if isinstance(team_id, str):
        try:
            team_id_int = int(team_id)
        except ValueError:
            pass
    
    if m_teams is not None:
        if team_id_int in m_teams['TeamID'].values or str(team_id) in m_teams['TeamID'].astype(str).values:
            return 'men'
    
    if w_teams is not None:
        if team_id_int in w_teams['TeamID'].values or str(team_id) in w_teams['TeamID'].astype(str).values:
            return 'women'
    
    # 如果无法确定，默认返回男队
    return 'men'


def generate_all_possible_matchups(data_dict, gender='both', max_teams=None):
    """
    生成所有可能的对阵组合
    
    参数:
        data_dict (dict): 数据字典
        gender (str): 'men', 'women', 或 'both'
        max_teams (int, optional): 限制处理的队伍数量
        
    返回:
        pd.DataFrame: 包含所有可能对阵的数据框
    """
    # 获取样本提交，用于验证ID格式
    sample_sub = data_dict.get('sample_sub')
    
    # 如果有样本提交，优先使用样本提交中的ID
    if sample_sub is not None:
        print(f"使用样本提交中的 {len(sample_sub)} 个ID")
        return sample_sub[['ID']].copy()
    
    # 以下是原始代码，只在没有样本提交时执行
    result = {}
    
    # 处理男子队伍
    if gender in ['men', 'both']:
        m_teams = data_dict.get('m_teams', None)
        if m_teams is not None:
            # 获取所有队伍ID并排序
            all_m_teams = sorted(m_teams['TeamID'].unique().tolist())
            
            # 生成所有组合，确保team1 < team2
            m_matchups = []
            for i, team1 in enumerate(all_m_teams):
                for team2 in all_m_teams[i+1:]:  # 只取更大的ID
                    m_matchups.append((team1, team2))
            
            print(f"生成了 {len(m_matchups)} 个男子队伍对阵组合")
            result['men'] = m_matchups
    
    # 处理女子队伍
    if gender in ['women', 'both']:
        w_teams = data_dict.get('w_teams', None)
        if w_teams is not None:
            # 获取所有队伍ID并排序
            all_w_teams = sorted(w_teams['TeamID'].unique().tolist())
            
            # 生成所有组合，确保team1 < team2
            w_matchups = []
            for i, team1 in enumerate(all_w_teams):
                for team2 in all_w_teams[i+1:]:  # 只取更大的ID
                    w_matchups.append((team1, team2))
            
            print(f"生成了 {len(w_matchups)} 个女子队伍对阵组合")
            result['women'] = w_matchups
    
    # 返回结果
    if gender == 'men':
        return result.get('men', [])
    elif gender == 'women':
        return result.get('women', [])
    else:
        return result


def batch_process_matchups(matchup_batch, features_dict, data_dict, model, model_columns, year, gender):
    """
    批量处理对阵预测，优化CPU利用和内存管理
    
    Parameters:
        matchup_batch (list): 待处理的对阵列表
        features_dict (dict): 特征数据字典
        data_dict (dict): 原始数据字典
        model: 预测模型
        model_columns: 模型所需列
        year (int): 预测年份
        gender (str): 'men'或'women'
    
    Returns:
        list: 处理结果列表
    """
    results = []
    for team1_id, team2_id in matchup_batch:
        # 确保team1_id < team2_id
        if team1_id > team2_id:
            team1_id, team2_id = team2_id, team1_id
            
        # 构建特征并预测
        matchup_features = build_matchup_features(
            team1_id, team2_id, features_dict, 
            target_year=year, round_num=2, gender=gender, data_dict=data_dict
        )
        
        # 处理可能的错误情况
        if matchup_features is None or len(matchup_features) == 0:
            results.append({
                'Team1': team1_id,
                'Team2': team2_id,
                'Pred': 0.5  # 默认预测值
            })
            continue
            
        # 确保特征格式正确
        if model_columns is not None:
            # 确保特征列顺序与模型期望的一致
            feature_values = []
            for col in model_columns:
                if col in matchup_features:
                    feature_values.append(matchup_features[col])
                else:
                    feature_values.append(0)  # 填充缺失特征
            
            # 转为numpy数组
            feature_values = np.array(feature_values).reshape(1, -1)
        else:
            feature_values = np.array(list(matchup_features.values())).reshape(1, -1)
        
        # 进行预测
        try:
            pred = model.predict_proba(feature_values)[0][1]
            
            # 添加微小随机扰动以避免预测值完全相同
            noise = np.random.uniform(-0.01, 0.01)
            pred = max(0.01, min(0.99, pred + noise))
            
            results.append({
                'Team1': team1_id,
                'Team2': team2_id,
                'Pred': pred
            })
        except Exception as e:
            print(f"预测错误 ({gender}): {e}")
            results.append({
                'Team1': team1_id,
                'Team2': team2_id,
                'Pred': 0.5  # 错误时使用默认值
            })
    
    return results


def worker_process_batches(worker_batch_list, features_dict, data_dict, model, model_columns, year, gender):
    """优化的工作进程函数，处理一批对阵"""
    results = []
    batch_size = len(worker_batch_list)
    
    # 添加进度指示器
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(worker_batch_list, desc=f"{gender}队伍对阵处理", leave=False)):
        batch_results = batch_process_matchups(
            batch, features_dict, data_dict, model, model_columns, year, gender
        )
        
        # 添加对批次结果的输出
        if i == 0:  # 只打印第一个批次的样本
            print(f"批次结果示例 ({gender}, 前5个):")
            for j, res in enumerate(batch_results[:5]):
                print(f"  {j+1}. Team1={res['Team1']}, Team2={res['Team2']}, Pred={res['Pred']:.4f}")
        
        results.extend(batch_results)
        
        # 添加定期进度报告
        if (i+1) % 10 == 0 or i == len(worker_batch_list) - 1:
            completion = (i+1) / len(worker_batch_list) * 100
            print(f"进度 ({gender}): {completion:.1f}% - 已完成 {i+1}/{len(worker_batch_list)} 批次")
            
    return results


def prepare_all_predictions(model, features_dict, data_dict, model_columns=None, year=2025, n_jobs=-1, gender='men'):
    """为所有可能的球队对阵准备预测结果，优化并行处理"""
    # 生成所有可能的对阵
    all_matchups = generate_all_possible_matchups(data_dict, gender=gender)
    
    print(f"为 {gender} 队伍生成 {len(all_matchups)} 个可能的对阵预测...")
    
    # 优化任务数量，避免使用过多CPU资源
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() - 1, 8)  # 限制最大并行数
    n_jobs = max(1, n_jobs)  # 确保至少有一个作业
    
    print(f"使用 {n_jobs} 个并行作业处理预测")
    
    # 分批处理，避免内存问题
    batch_size = calculate_optimal_batch_size(len(all_matchups), n_jobs)
    batched_matchups = []
    
    for i in range(0, len(all_matchups), batch_size):
        batch = all_matchups[i:i+batch_size]
        batched_matchups.append(batch)
    
    # 将批次分配给工作进程
    worker_batches = []
    for i in range(n_jobs):
        worker_batches.append([])
    
    # 均匀分配批次，确保负载均衡
    for i, batch in enumerate(batched_matchups):
        worker_idx = i % n_jobs
        worker_batches[worker_idx].append(batch)
    
    # 使用并行处理处理每个工作进程的批次
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, timeout=3600)(
            delayed(worker_process_batches)(
                worker_batch, features_dict, data_dict, model, model_columns, year, gender
            )
            for worker_batch in worker_batches if worker_batch  # 跳过空批次
        )
        
        # 展平结果
        all_results = []
        for res_list in results:
            all_results.extend(res_list)
            
    except Exception as e:
        print(f"并行处理错误: {e}")
        print("回退到串行处理...")
        
        # 回退到串行处理
        all_results = []
        for worker_batch in worker_batches:
            if not worker_batch:
                continue
            batch_results = worker_process_batches(
                worker_batch, features_dict, data_dict, model, model_columns, year, gender
            )
            all_results.extend(batch_results)
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(all_results)
    
    # 确保列类型正确
    predictions_df['Team1'] = predictions_df['Team1'].astype(int)
    predictions_df['Team2'] = predictions_df['Team2'].astype(int)
    predictions_df['Pred'] = predictions_df['Pred'].astype(float)
    
    # 修改：添加ID列 - 安全地处理可能是字符串的值
    def safe_int_convert(val):
        """更强大的整数转换函数，处理各种ID格式"""
        try:
            # 如果是纯数字，直接转换
            return int(val)
        except (ValueError, TypeError):
            # 可能是带前缀的ID或其他特殊格式
            if isinstance(val, str):
                # 先尝试移除所有非数字字符
                import re
                nums = re.findall(r'\d+', val)
                if nums:
                    return int(nums[0])
                
                # 特殊情况：如果是"Seed_X"格式
                if val.startswith("Seed_"):
                    try:
                        return int(val.split("_")[1])
                    except:
                        pass
                    
        # 如果无法转换，返回一个有效默认值
        return 1001  # 使用一个不太可能与实际ID冲突的值
            
    predictions_df['ID'] = predictions_df.apply(
        lambda row: f"{year}_{safe_int_convert(row['Team1'])}_{safe_int_convert(row['Team2'])}", axis=1
    )
    
    # 检查ID格式
    print(f"生成的ID示例 ({gender}):")
    for i, id_val in enumerate(predictions_df['ID'].head(5).values):
        print(f"  {i+1}. {id_val}")
    
    # 检查预测值的分布
    if len(predictions_df) > 0:
        print(f"预测值统计 ({gender}):")
        print(f"  最小值: {predictions_df['Pred'].min():.4f}")
        print(f"  最大值: {predictions_df['Pred'].max():.4f}")
        print(f"  平均值: {predictions_df['Pred'].mean():.4f}")
        print(f"  标准差: {predictions_df['Pred'].std():.4f}")
    
    print(f"已完成 {len(predictions_df)} 个对阵的预测")
    
    # 校准预测值
    predictions_df = apply_prediction_calibration(predictions_df, gender)
    
    return predictions_df


def create_calibrated_submission(predictions_df, sample_submission, spline_model=None, 
                          data_dict=None, filename=None):
    """
    创建与样本提交格式一致的校准提交文件
    
    参数:
        predictions_df (pd.DataFrame): 包含所有预测的数据框
        sample_submission (pd.DataFrame): 样本提交文件
        spline_model (callable, optional): 校准模型函数
        data_dict (dict, optional): 数据字典
        filename (str, optional): 输出文件名
        
    返回:
        pd.DataFrame: 校准后的提交数据框
    """
    # 确保ID格式与样本提交完全一致
    # 有些ID可能需要添加性别标识，如'_men'或'_women'
    
    # 分析样本提交中ID的格式
    id_example = sample_submission['ID'].iloc[0] if len(sample_submission) > 0 else ""
    has_gender_suffix = ('_men' in id_example or '_women' in id_example)
    
    # 如果样本中ID包含性别后缀，但预测结果中没有，添加它
    if has_gender_suffix and not any('_men' in id or '_women' in id for id in predictions_df['ID']):
        # 根据identify_team_gender函数添加正确的性别后缀
        def add_gender_suffix(row):
            team1_id, team2_id = row['ID'].split('_')[1:3]
            team1_gender = identify_team_gender(int(team1_id), data_dict)
            return f"{row['ID']}_{team1_gender}"
        
        predictions_df['ID'] = predictions_df.apply(add_gender_suffix, axis=1)
    
    # 创建样本提交的ID索引字典用于匹配
    sample_ids = set(sample_submission['ID'].values)
    
    print(f"样本提交包含 {len(sample_ids)} 个ID")
    print(f"预测数据包含 {len(predictions_df)} 行")
    
    # 过滤预测结果，只保留样本提交中包含的ID
    filtered_predictions = predictions_df[predictions_df['ID'].isin(sample_ids)].copy()
    
    print(f"过滤后的预测数据包含 {len(filtered_predictions)} 行")
    
    # 检查是否有缺失的ID
    missing_ids = sample_ids - set(filtered_predictions['ID'].values)
    if missing_ids:
        print(f"警告: 有 {len(missing_ids)} 个ID在预测中缺失")
        # 可以为缺失的ID填充0.5的预测值
        missing_df = pd.DataFrame({
            'ID': list(missing_ids),
            'Pred': [0.5] * len(missing_ids)
        })
        filtered_predictions = pd.concat([filtered_predictions, missing_df], ignore_index=True)
    
    # 应用校准模型（如果提供）
    if spline_model is not None:
        filtered_predictions['Pred'] = filtered_predictions['Pred'].apply(spline_model)
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'ID': filtered_predictions['ID'],
        'Pred': filtered_predictions['Pred']
    })
    
    # 确保行数与样本提交一致
    if len(submission_df) != len(sample_submission):
        print(f"警告: 提交文件包含 {len(submission_df)} 行，样本提交包含 {len(sample_submission)} 行")
        
        # 如果有额外行，根据样本提交筛选
        if len(submission_df) > len(sample_submission):
            submission_df = submission_df[submission_df['ID'].isin(sample_submission['ID'])]
            
        # 如果有缺失行，添加缺失ID（填充0.5）
        if len(submission_df) < len(sample_submission):
            missing_ids = set(sample_submission['ID']) - set(submission_df['ID'])
            missing_df = pd.DataFrame({
                'ID': list(missing_ids),
                'Pred': [0.5] * len(missing_ids)
            })
            submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
    
    # 确保与样本提交顺序一致
    submission_df = submission_df.set_index('ID').loc[sample_submission['ID']].reset_index()
    
    # 保存文件
    if filename:
        submission_df.to_csv(filename, index=False)
        print(f"校准后的提交文件已保存至 {filename}")
        print(f"最终提交文件包含 {len(submission_df)} 行")
    
    return submission_df


def validate_submission(submission_df, data_dict):
    """
    验证提交文件的格式是否正确
    
    参数:
        submission_df (pd.DataFrame): 要验证的提交数据框
        data_dict (dict): 数据字典
        
    返回:
        bool: 验证是否通过
    """
    # 检查提交文件的列
    if 'ID' not in submission_df.columns or 'Pred' not in submission_df.columns:
        print("错误: 提交文件必须包含 'ID' 和 'Pred' 列")
        return False
    
    # 检查样本提交格式
    sample_submission = data_dict.get('sample_sub')
    if sample_submission is None:
        print("警告: 未找到样本提交文件，无法完全验证格式")
    else:
        # 检查行数
        if len(submission_df) != len(sample_submission):
            print(f"警告: 提交文件包含 {len(submission_df)} 行，样本提交包含 {len(sample_submission)} 行")
            
            # 分析多出的ID
            if len(submission_df) > len(sample_submission):
                extra_ids = set(submission_df['ID']) - set(sample_submission['ID'])
                print(f"发现 {len(extra_ids)} 个多余ID，需要移除")
            
            # 分析缺失的ID
            missing_ids = set(sample_submission['ID']) - set(submission_df['ID'])
            if missing_ids:
                print(f"缺少 {len(missing_ids)} 个ID，需要添加")
            
            return False
        
        # 检查ID顺序
        if not submission_df['ID'].equals(sample_submission['ID']):
            print("警告: 提交文件的ID顺序与样本提交不一致")
            return False
    
    # 检查预测值的范围
    if submission_df['Pred'].min() < 0 or submission_df['Pred'].max() > 1:
        print(f"错误: 预测值必须在[0,1]范围内，当前范围: [{submission_df['Pred'].min()}, {submission_df['Pred'].max()}]")
        return False
    
    print("提交文件验证通过!")
    return True


@lru_cache(maxsize=10000)
def cached_build_matchup_features(team1, team2, target_year, round_num, gender):
    """缓存版本的特征构建函数，避免重复计算"""
    # 这个函数需要修改原始函数来使用，因为lru_cache需要可哈希参数
    # ...

def calculate_optimal_batch_size(total_items, n_jobs, available_memory_mb=None):
    """动态计算最优批次大小，根据数据量、作业数和可用内存"""
    # 如果未指定可用内存，尝试获取系统内存
    if available_memory_mb is None:
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # 无法获取系统内存时使用默认值
            available_memory_mb = 2000  # 默认2GB
    
    # 每项目估计内存使用量(MB)
    est_memory_per_item = 0.01  # 假设每个对阵项占用10KB
    
    # 根据可用内存和作业数计算安全的批次大小
    memory_safe_batch = int(available_memory_mb / (est_memory_per_item * n_jobs * 1.5))
    
    # 基本批次大小
    base_size = 100
    
    # 根据数据量调整
    if total_items > 10000:
        base_size = 200
    elif total_items > 5000:
        base_size = 150
    
    # 取两种方法计算结果的较小值，确保内存安全
    optimal_size = min(base_size, memory_safe_batch)
    
    # 确保每个作业至少有一个批次
    return max(optimal_size, total_items // (n_jobs * 10))

def apply_prediction_calibration(predictions_df, gender):
    """校准预测值，保留原始分布形状但调整均值到合理范围"""
    # 确保所有预测值在[0.01, 0.99]范围内
    predictions_df['Pred'] = predictions_df['Pred'].clip(0.01, 0.99)
    
    # 如果是男队且预测值过低，或女队预测值过高，进行校准
    if gender == 'men' and predictions_df['Pred'].mean() < 0.45:
        print(f"校准{gender}队伍预测值 (当前平均: {predictions_df['Pred'].mean():.4f})")
        # 使用均值移位而非范围压缩，保留原始分布形状
        current_mean = predictions_df['Pred'].mean()
        shift = 0.5 - current_mean  # 移动到0.5均值
        predictions_df['Pred'] = predictions_df['Pred'] + shift
        # 确保所有值在[0.01, 0.99]范围内
        predictions_df['Pred'] = predictions_df['Pred'].clip(0.01, 0.99)
        print(f"校准后平均: {predictions_df['Pred'].mean():.4f}")
    elif gender == 'women' and predictions_df['Pred'].mean() > 0.55:
        print(f"校准{gender}队伍预测值 (当前平均: {predictions_df['Pred'].mean():.4f})")
        # 使用均值移位而非范围压缩
        current_mean = predictions_df['Pred'].mean()
        shift = 0.5 - current_mean
        predictions_df['Pred'] = predictions_df['Pred'] + shift
        # 确保所有值在[0.01, 0.99]范围内
        predictions_df['Pred'] = predictions_df['Pred'].clip(0.01, 0.99)
        print(f"校准后平均: {predictions_df['Pred'].mean():.4f}")
    
    return predictions_df

# 添加最终验证函数
def validate_final_submission(submission_file):
    """更严格的提交验证，确保完全匹配比赛要求"""
    try:
        # 加载提交文件
        submission = pd.read_csv(submission_file)
        print(f"提交文件包含 {len(submission)} 行")
        
        # 检查列名
        if not all(col in submission.columns for col in ['ID', 'Pred']):
            print("错误: 提交文件必须包含'ID'和'Pred'列")
            return False
        
        # 检查ID格式和规范性
        # 1. 所有ID必须以'2025_'开头
        year_check = all(str(id_str).startswith('2025_') for id_str in submission['ID'])
        if not year_check:
            # 详细列出错误的ID
            non_2025_ids = [id_str for id_str in submission['ID'] if not str(id_str).startswith('2025_')]
            print(f"错误: 发现{len(non_2025_ids)}个不以'2025_'开头的ID")
            if len(non_2025_ids) > 0:
                print(f"示例: {non_2025_ids[:5]}")
                
            # 尝试修复
            print("尝试修复ID格式...")
            submission['ID'] = submission['ID'].apply(
                lambda x: f"2025_{x.split('_', 1)[1]}" if isinstance(x, str) and '_' in x and not x.startswith('2025_') else x
            )
            # 再次检查
            non_2025_ids = [id_str for id_str in submission['ID'] if not str(id_str).startswith('2025_')]
            if len(non_2025_ids) > 0:
                print(f"修复后仍有{len(non_2025_ids)}个格式不正确的ID")
                return False
            else:
                print("ID格式已修复")
        
        # 2. 检查是否有重复ID
        duplicates = submission['ID'].duplicated()
        if duplicates.any():
            print(f"错误: 发现{duplicates.sum()}个重复ID")
            return False
            
        # 检查预测值范围
        pred_range_valid = (submission['Pred'] >= 0).all() and (submission['Pred'] <= 1).all()
        if not pred_range_valid:
            # 修复超出范围的预测值
            print("修复超出范围的预测值...")
            submission['Pred'] = submission['Pred'].clip(0, 1)
        
        # 显示预测值分布统计
        print(f"预测值分布: 最小={submission['Pred'].min():.4f}, 平均={submission['Pred'].mean():.4f}, 最大={submission['Pred'].max():.4f}")
        
        # 保存修复后的文件
        if not year_check or not pred_range_valid:
            fixed_path = submission_file.replace('.csv', '_fixed.csv')
            submission.to_csv(fixed_path, index=False)
            print(f"修复后的提交文件已保存至: {fixed_path}")
            
        print("✓ 提交文件验证通过！符合2025比赛要求")
        return True
    
    except Exception as e:
        print(f"验证提交文件时出错: {e}")
        return False
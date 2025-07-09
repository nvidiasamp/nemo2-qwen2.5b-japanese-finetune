#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Model Evaluation

This module handles evaluation and optimization of model predictions.
本模块处理模型预测的评估和优化。

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
Version: 2.1 ### 增加了预测所有可能的球队对阵的假设结果
Version: 3.0 ### 增加了cudf的支持
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from typing import Dict, Optional, Tuple, Union, List, Any
import cupy as cp
from contextlib import contextmanager
from utils import gpu_context, to_gpu, to_cpu
from scipy.interpolate import UnivariateSpline


def apply_brier_optimal_strategy(predictions_df: pd.DataFrame, 
                               lower_bound: float = 0.3, 
                               upper_bound: float = 0.36,
                               adjustment_factor: float = 0.5) -> pd.DataFrame:
    """
    Apply optimal risk strategy under Brier score.
    
    根据Brier分数应用最优风险策略。
    
    According to theoretical analysis, modifying predictions within certain
    probability ranges can optimize the expected Brier score by applying
    strategic risk adjustments.
    
    根据理论分析，对特定概率范围内的预测进行修改可以通过应用战略风险调整来优化预期的Brier分数。
    
    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        DataFrame containing prediction probabilities
        包含预测概率的数据框
    lower_bound : float, optional (default=0.3)
        Lower threshold for applying risk strategy
        应用风险策略的下限阈值
    upper_bound : float, optional (default=0.36)
        Upper threshold for applying risk strategy
        应用风险策略的上限阈值
    adjustment_factor : float, optional (default=0.5)
        Factor used to adjust predictions in the risk range
        用于调整风险范围内预测的因子
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with adjusted prediction probabilities
        调整后预测概率的数据框
    """
    # Gold medal solution proves: theoretically, taking a risk strategy for games with 33.3% win probability is optimal
    # Max value of f(p) = p(1-p)^2 occurs at p=1/3
    # 金牌解决方案证明：理论上，对于胜率为33.3%的比赛采取风险策略是最优的
    # 函数f(p) = p(1-p)^2的最大值出现在p=1/3处
    
    # 创建掩码而不是Series视图
    risky_mask = (predictions_df['Pred'] >= lower_bound) & (predictions_df['Pred'] <= upper_bound)
    risky_count = risky_mask.sum()
    
    if risky_count > 0:
        # 只复制需要修改的部分
        predictions_copy = predictions_df.copy()
        
        # 直接使用布尔索引修改
        risky_indices = predictions_copy.index[risky_mask]
        current_preds = predictions_copy.loc[risky_indices, 'Pred']
        predictions_copy.loc[risky_indices, 'Pred'] = 0.5 + (current_preds - lower_bound) * adjustment_factor
        
        # 生成报告
        total_predictions = len(predictions_copy)
        print(f"Applying optimal risk strategy:")
        print(f"  - Total predictions: {total_predictions}")
        print(f"  - Predictions with risk strategy applied: {risky_count} ({risky_count/total_predictions*100:.2f}%)")
        print(f"  - Adjustment range: [{lower_bound}, {upper_bound}] with factor {adjustment_factor}")
        
        return predictions_copy
    else:
        # 没有修改时返回原始数据
        print("No predictions in the risk range - no adjustments applied")
        return predictions_df


def evaluate_predictions(y_true, y_pred, confidence_thresholds=(0.3, 0.7),
                       gender=None, use_gpu=False):  # 默认关闭GPU避免问题
    """评估预测结果，更稳健的实现"""
    # 确保数据类型正确
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    
    # 检查输入数据
    if len(y_true) != len(y_pred):
        print(f"警告: 真实值和预测值长度不匹配 ({len(y_true)} vs {len(y_pred)})")
        return {"brier_score": 1.0, "log_loss": 1.0, "roc_auc": 0.5}
    
    # 检查标签是否只有一个类别
    unique_true = np.unique(y_true)
    if len(unique_true) <= 1:
        print(f"警告: 真实标签只有一个类别，无法计算有意义的ROC AUC")
        roc_auc_val = 0.5  # 使用中性值
    else:
        # 使用CPU计算ROC AUC
        try:
            roc_auc_val = roc_auc_score(y_true, y_pred)
        except Exception as e:
            print(f"ROC AUC计算失败: {e}，使用默认值0.5")
            roc_auc_val = 0.5
    
    # 计算Brier分数
    brier_val = brier_score_loss(y_true, y_pred)
    
    # 计算对数损失，确保值在有效范围内
    y_pred_clipped = np.clip(y_pred, 1e-15, 1-1e-15)
    log_loss_val = log_loss(y_true, y_pred_clipped)
    
    metrics = {
        "brier_score": brier_val,
        "log_loss": log_loss_val,
        "roc_auc": roc_auc_val
    }
    
    return metrics


def visualization_prediction_distribution(y_pred: Union[List, np.ndarray], 
                                         y_true: Optional[Union[List, np.ndarray]] = None, 
                                         save_path: Optional[str] = None,
                                         show_plot: bool = True,
                                         fig_size: Tuple[int, int] = (12, 6),
                                         key_points: List[float] = [0.3, 0.5, 0.7],
                                         title: Optional[str] = None) -> plt.Figure:
    """
    可视化预测分布。
    
    Parameters:
    -----------
    y_pred : array-like
        预测的概率值，范围在[0, 1]之间
    y_true : array-like, optional
        真实的二元标签（0或1）
    save_path : str, optional
        保存可视化结果的路径
    show_plot : bool, optional (default=True)
        是否显示图形（批处理时设为False）
    fig_size : tuple, optional (default=(12, 6))
        图形大小
    key_points : list, optional (default=[0.3, 0.5, 0.7])
        在可视化中显示的关键概率点
    title : str, optional
        图表标题
        
    Returns:
    --------
    matplotlib.figure.Figure
        生成的图形对象
    """
    # 确保输入不为空且类型正确
    if y_pred is None or len(y_pred) == 0:
        print("警告：预测数据为空，无法生成分布图")
        # 返回空图形
        fig = plt.figure(figsize=fig_size)
        plt.title("无数据可显示")
        return fig
    
    # 转换为numpy数组并过滤无效值
    y_pred_np = np.asarray(y_pred)
    valid_mask = ~np.isnan(y_pred_np)
    y_pred_np = y_pred_np[valid_mask]  # 移除NaN值
    
    if len(y_pred_np) == 0:
        print("警告：过滤NaN后预测数据为空，无法生成分布图")
        fig = plt.figure(figsize=fig_size)
        plt.title("过滤后无有效数据")
        return fig
    
    # 输入验证 - 确保值在有效范围内
    if not (0 <= np.min(y_pred_np) and np.max(y_pred_np) <= 1):
        print(f"警告：预测值不在[0,1]范围内（{np.min(y_pred_np):.4f}, {np.max(y_pred_np):.4f}），进行截断")
        y_pred_np = np.clip(y_pred_np, 0, 1)
    
    if y_true is not None:
        y_true_np = np.asarray(y_true)
        # 确保大小匹配
        if len(y_true_np) != len(y_pred):
            print(f"警告：真实标签和预测值长度不匹配 ({len(y_true_np)} vs {len(y_pred)})，忽略真实标签")
            y_true = None
        else:
            # 应用相同的NaN过滤
            y_true_np = y_true_np[valid_mask]
            # 验证真实值是否为二元值
            if not np.all(np.isin(y_true_np, [0, 1])):
                print("警告：真实标签包含非二元值(0/1)，将忽略真实标签进行可视化。")
                y_true = None
    
    # 创建指定大小的图形
    fig = plt.figure(figsize=fig_size)
    
    # 绘制预测分布
    ax1 = plt.subplot(1, 2 if y_true is not None else 1, 1)
    sns.histplot(y_pred_np, bins=20, kde=True, ax=ax1)
    ax1.set_title('预测分布')
    ax1.set_xlabel('获胜预测概率')
    ax1.set_ylabel('频率')
    
    # 添加主标题
    if title:
        plt.suptitle(title, fontsize=16, y=1.05)
    
    # 添加关键概率点的垂直线
    colors = ['r', 'g', 'r']  # 默认颜色
    if len(key_points) != len(colors):
        # 动态生成颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(key_points)))
    
    for i, point in enumerate(key_points):
        if 0 <= point <= 1:  # 确保点在有效范围内
            ax1.axvline(point, color=colors[i % len(colors)], linestyle='--', 
                       alpha=0.5, label=f'{point}')
    ax1.legend()
    
    # 计算摘要统计信息
    try:
        mean_pred = np.mean(y_pred_np)
        median_pred = np.median(y_pred_np)
        std_pred = np.std(y_pred_np)
        
        # 添加统计信息文本
        stats_text = f"均值: {mean_pred:.3f}\n中位数: {median_pred:.3f}\n标准差: {std_pred:.3f}"
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    except Exception as e:
        print(f"计算统计信息时出错: {e}")
    
    # 如果提供了真实值，绘制按结果分组的预测
    if y_true is not None:
        try:
            ax2 = plt.subplot(1, 2, 2)
            
            # 创建数据框用于绘图
            plot_data = {'y_true': y_true_np, 'y_pred': y_pred_np}
            df = pd.DataFrame(plot_data)
            
            # 绘制箱线图
            sns.boxplot(x='y_true', y='y_pred', data=df, ax=ax2)
            
            # 添加散点
            sns.stripplot(x='y_true', y='y_pred', data=df, 
                         size=3, color='black', alpha=0.2, jitter=True, ax=ax2)
            
            ax2.set_title('按实际结果分组的预测概率')
            ax2.set_xlabel('实际结果 (0=输, 1=赢)')
            ax2.set_ylabel('获胜预测概率')
            
            # 计算按结果分组的统计信息
            for outcome in df['y_true'].unique():
                outcome_preds = df[df['y_true'] == outcome]['y_pred']
                if len(outcome_preds) > 0:
                    mean_val = outcome_preds.mean()
                    ax2.text(outcome, 0.02, f'均值: {mean_val:.3f}', 
                            ha='center', bbox=dict(boxstyle='round', alpha=0.1))
        except Exception as e:
            print(f"绘制按结果分组的预测时出错: {e}")
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")
        except Exception as e:
            print(f"保存图表时出错: {e}")
    
    # 显示图形
    if show_plot:
        plt.show()
    
    return fig


def calibration_curve(y_true: Union[List, np.ndarray], 
                     y_pred: Union[List, np.ndarray], 
                     n_bins: int = 10,
                     save_path: Optional[str] = None,
                     show_plot: bool = True,
                     title: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate and visualize prediction calibration curve.
    
    生成并可视化预测校准曲线。
    
    A calibration curve plots predicted probabilities against actual outcome rates.
    A well-calibrated model should produce points that lie close to the diagonal.
    
    校准曲线绘制预测概率与实际结果率之间的关系。校准良好的模型应该产生接近对角线的点。
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
        真实的二元标签（0或1）
    y_pred : array-like
        Predicted probabilities in range [0, 1]
        预测的概率值，范围在[0, 1]之间
    n_bins : int, optional (default=10)
        Number of bins to use for the calibration curve
        用于校准曲线的分箱数量
    save_path : str, optional
        Path to save the visualization
        保存可视化结果的路径
    show_plot : bool, optional (default=True)
        Whether to display the plot (set to False for batch processing)
        是否显示图形（批处理时设为False）
    title : str, optional (default=None)
        Title for the plot
        图表标题
        
    Returns:
    --------
    dict
        Dictionary containing calibration data:
        包含校准数据的字典：
        'bin_centers': Centers of the probability bins
                       概率分箱的中心
        'bin_actual': Actual outcome rate for each bin
                     每个分箱的实际结果率
        'bin_counts': Number of predictions in each bin
                     每个分箱中的预测数量
        'bin_errors': Standard error for each bin's estimate
                     每个分箱估计的标准误差
    """
    # Convert inputs to numpy arrays
    # 将输入转换为numpy数组
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Input validation
    # 输入验证
    if not (0 <= np.min(y_pred_np) and np.max(y_pred_np) <= 1):
        raise ValueError("Predicted probabilities must be in the range [0, 1]")
    
    if not np.all(np.isin(y_true_np, [0, 1])):
        raise ValueError("True labels must be binary (0 or 1)")
    
    if n_bins < 2:
        raise ValueError("Number of bins must be at least 2")
    
    # Create bin edges and calculate bin centers
    # 创建分箱边界并计算分箱中心
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Use vectorized operations to calculate bin indices for each prediction
    # 使用向量化操作计算每个预测的分箱索引
    bin_indices = np.digitize(y_pred_np, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure indices don't exceed bins
    
    # Initialize arrays for bin statistics
    # 初始化分箱统计数组
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_errors = np.zeros(n_bins)
    
    # Calculate bin sums and counts using vectorized operations
    # 使用向量化操作计算分箱总和和计数
    for i in range(n_bins):
        mask = (bin_indices == i)
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_sums[i] = np.sum(y_true_np[mask])
    
    # Calculate actual outcome rates and standard errors
    # 计算实际结果率和标准误差
    bin_actual = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_actual[i] = bin_sums[i] / bin_counts[i]
            # Calculate standard error using binomial formula √(p*(1-p)/n)
            # 使用二项式公式√(p*(1-p)/n)计算标准误差
            if bin_counts[i] > 1:  # Avoid division by zero
                bin_errors[i] = np.sqrt(bin_actual[i] * (1 - bin_actual[i]) / bin_counts[i])
    
    fig = plt.figure(figsize=(10, 6))
    
    # Plot calibration curve with error bars
    # 绘制带误差线的校准曲线
    plt.errorbar(bin_centers, bin_actual, yerr=bin_errors, fmt='o-', 
                label='Calibration Curve', ecolor='lightgray', capsize=3)
    
    # Add perfect calibration line
    # 添加完美校准线
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    # Add bin counts as text
    # 添加分箱计数为文本
    for i in range(n_bins):
        if bin_counts[i] > 0:  # Only add text for bins with data
            plt.text(bin_centers[i], bin_actual[i] + max(0.03, bin_errors[i] + 0.01), 
                    f'n={int(bin_counts[i])}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Calculate overall calibration error metrics
    # 计算总体校准误差指标
    mce = np.sum(np.abs(bin_actual - bin_centers) * (bin_counts / np.sum(bin_counts)))
    rmsce = np.sqrt(np.sum(np.square(bin_actual - bin_centers) * (bin_counts / np.sum(bin_counts))))
    
    # Add calibration error metrics to the plot
    # 将校准误差指标添加到图中
    plt.text(0.05, 0.95, f'MCE: {mce:.4f}\nRMSCE: {rmsce:.4f}', 
            transform=plt.gca().transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Outcome Rate')
    plt.title(title or 'Prediction Calibration Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure if path is provided
    # 如果提供了路径，则保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Only show the plot if requested
    # 仅在请求时显示图形
    if show_plot:
        plt.show()
    
    # Return calibration data
    # 返回校准数据
    return {
        'bin_centers': bin_centers,
        'bin_actual': bin_actual,
        'bin_counts': bin_counts,
        'bin_errors': bin_errors,
        'metrics': {
            'mce': mce,  # Mean Calibration Error
            'rmsce': rmsce  # Root Mean Squared Calibration Error
        }
    }


def create_spline_calibration_model(y_pred, y_true, visualize=False, save_path=None):
    """
    创建将预测得分差异转换为胜率的样条校准模型
    
    参数:
        y_pred (array-like): 预测的得分差异
        y_true (array-like): 真实标签（1表示胜利，0表示失败）
        visualize (bool): 是否可视化样条函数
        save_path (str): 可视化保存路径
        
    返回:
        callable: 校准样条函数
    """
    # 确保输入是numpy数组
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # 创建数据对并按预测值排序
    data = list(zip(y_pred, np.where(y_true > 0, 1, 0)))
    data = sorted(data, key=lambda x: x[0])
    
    # 创建样条拟合的输入
    x_values = [item[0] for item in data]
    y_values = [item[1] for item in data]
    
    # 使用唯一值创建字典，避免重复输入点
    data_dict = {}
    for x, y in data:
        if x not in data_dict:
            data_dict[x] = []
        data_dict[x].append(y)
    
    # 对每个x值取平均y值
    x_unique = list(data_dict.keys())
    y_unique = [np.mean(data_dict[x]) for x in x_unique]
    
    # 创建样条模型
    spline_model = UnivariateSpline(x_unique, y_unique, s=0.1)  # s参数控制平滑度
    
    # 限制输出范围在[0.025, 0.975]之间
    def calibrated_predict(x):
        return np.clip(spline_model(np.clip(x, -30, 30)), 0.025, 0.975)
    
    # 可视化
    if visualize:
        # 创建可视化数据框
        plot_df = pd.DataFrame({"pred": y_pred, "label": np.where(y_true > 0, 1, 0), 
                              "spline": calibrated_predict(y_pred)})
        plot_df["pred_int"] = plot_df["pred"].astype(int)
        plot_df = plot_df.groupby('pred_int').mean().reset_index()
        
        # 绘制图表
        plt.figure(figsize=[10, 6])
        plt.plot(plot_df.pred_int, plot_df.spline, label='样条校准')
        plt.plot(plot_df.pred_int, plot_df.label, label='实际胜率')
        plt.xlabel('预测得分差异')
        plt.ylabel('胜率')
        plt.title('样条校准曲线')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    return calibrated_predict


def analyze_538_feature_importance(model, feature_names, plot=True):
    """
    专门分析538特征的重要性
    
    参数:
        model: 训练好的模型
        feature_names: 特征名称列表
        plot: 是否绘制图表
        
    返回:
        538特征重要性数据框
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # 提取538相关特征
    feature_538_indices = [i for i, name in enumerate(feature_names) if '538' in name]
    
    if not feature_538_indices:
        print("模型中没有538相关特征")
        return None
    
    # 创建538特征重要性数据框
    importance_538 = pd.DataFrame({
        'Feature': [feature_names[i] for i in feature_538_indices],
        'Importance': [model.feature_importances_[i] for i in feature_538_indices]
    }).sort_values('Importance', ascending=False)
    
    # 打印538特征重要性
    print("538特征重要性:")
    for i, (_, row) in enumerate(importance_538.iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.6f}")
    
    # 绘制538特征重要性
    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_538)
        plt.title('538特征重要性分析')
        plt.tight_layout()
        plt.show()
    
    return importance_538
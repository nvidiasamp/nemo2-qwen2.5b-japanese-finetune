#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Utility Functions

This module contains helper functions and common utilities for the prediction model.

Author: Junming Zhao
Date: 2025-03-16
Version: 2.0 ### 增加了针对女队的预测
Version: 2.1 ### 增加了预测所有可能的球队对阵的假设结果
Version: 3.0 ### 增加了cudf的支持
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import concurrent.futures
import joblib  # 添加 joblib 导入
import multiprocessing
import sys
import cupy as cp

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False
    print("cudf not available, DataFrame GPU acceleration disabled")

from contextlib import contextmanager


def set_random_seed(seed=42, use_gpu=True):
    """Set random seeds for reproducibility (with GPU support)"""
    random.seed(seed)
    np.random.seed(seed)
    if use_gpu:
        try:
            cp.random.seed(seed)
        except Exception as e:
            print(f"GPU随机种子设置失败: {e}")
    print(f"Random seed set to {seed}")


@contextmanager
def gpu_context(use_gpu=True):
    """改进的GPU上下文管理器，添加更好的错误处理和回退机制"""
    if not use_gpu:
        try:
            yield None
        finally:
            pass
        return
    
    # 检查是否有可用GPU
    gpu_available = False
    try:
        import cupy as cp
        gpu_available = True
        # 尝试获取GPU内存信息以验证可用性
        _ = cp.cuda.runtime.memGetInfo()
    except Exception as e:
        print(f"GPU初始化失败: {e}，将使用CPU")
        gpu_available = False
    
    if gpu_available:
        try:
            # 配置cupy使用合理的内存池限制
            try:
                cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            except Exception:
                # 如果managed memory不可用，使用标准内存池
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            
            print("GPU上下文启动：使用CUDA加速")
            yield True
        except Exception as e:
            print(f"GPU使用期间发生错误: {e}，切换到CPU")
            yield False
        finally:
            try:
                # 释放所有未使用的内存
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
    else:
        try:
            print("GPU不可用，使用CPU")
            yield False
        finally:
            pass


def to_gpu(data, use_gpu=True):
    """改进的GPU数据转换函数，支持批处理和内存优化"""
    if not use_gpu:
        return data
    
    try:
        # 检查GPU是否可用
        with gpu_context(use_gpu) as gpu_available:
            if not gpu_available:
                return data
                
        # 批量处理大型数据
        if isinstance(data, np.ndarray):
            if data.size > 1e7:  # 大于10M的数组
                # 分批处理
                batch_size = int(1e7)
                num_batches = (data.size + batch_size - 1) // batch_size
                gpu_data = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, data.size)
                    batch = cp.array(data[start_idx:end_idx])
                    gpu_data.append(batch)
                return cp.concatenate(gpu_data)
            return cp.array(data)
        elif isinstance(data, pd.DataFrame):
            if HAS_CUDF:
                if data.memory_usage().sum() > 1e8:  # 大于100MB的DataFrame
                    # 分批处理列
                    gpu_df = cudf.DataFrame()
                    for col in data.columns:
                        gpu_df[col] = cudf.Series(data[col])
                    return gpu_df
                return cudf.DataFrame(data)
            return data
        elif isinstance(data, pd.Series):
            if HAS_CUDF:
                return cudf.Series(data)
            return data
        return data
    except Exception as e:
        print(f"GPU数据转换失败: {e}")
        return data


def to_cpu(data):
    """将数据转移回CPU"""
    if isinstance(data, (cp.ndarray, cudf.DataFrame, cudf.Series)):
        try:
            return data.get() if isinstance(data, cp.ndarray) else data.to_pandas()
        except Exception as e:
            print(f"CPU数据转换失败: {e}")
    return data


def save_model(model, filename, model_columns=None, use_gpu=True):
    """Save a trained model to disk with compression (GPU support)"""
    try:
        model_data = {
            'model': model,
            'columns': model_columns,
            'gpu_enabled': use_gpu
        }
        joblib.dump(model_data, filename, compress=3)
        print(f"Model saved to {filename}")
        if model_columns:
            print(f"Saved with {len(model_columns)} feature columns")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False


def load_model(filename):
    """
    Load a trained model from disk
    从磁盘加载训练好的模型
    """
    try:
        # 使用joblib加载模型
        model_data = joblib.load(filename)
        
        # 如果是新格式（字典），则提取模型和列
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            columns = model_data.get('columns', None)
            print(f"模型已从 {filename} 加载")
            if columns:
                print(f"已加载 {len(columns)} 个特征列")
            return model, columns
        # 如果是旧格式（只有模型），则只返回模型
        else:
            print(f"模型已从 {filename} 加载（旧格式，无列名）")
            return model_data, None
            
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None


def save_features(features_dict, filename):
    """
    Save features dictionary to disk with compression
    将特征字典保存到磁盘并进行压缩
    """
    try:
        # 使用joblib保存并压缩特征
        # Use joblib to save and compress features
        joblib.dump(features_dict, filename, compress=3)
        print(f"Features saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving features: {str(e)}")
        return False


def load_features(filename):
    """
    Load features dictionary from disk
    从磁盘加载特征字典
    """
    try:
        # 使用joblib加载特征
        # Use joblib to load features
        features_dict = joblib.load(filename)
        print(f"Features loaded from {filename}")
        return features_dict
    except Exception as e:
        print(f"Error loading features: {str(e)}")
        return None


def timer(func):
    """
    Decorator to time function execution and monitor memory usage
    装饰器，用于计时函数执行并监控内存使用
    """
    def wrapper(*args, **kwargs):
        import time
        import psutil
        
        # 获取当前进程
        # Get current process
        process = psutil.Process(os.getpid())
        
        # 记录开始时间和内存
        # Record start time and memory
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Starting {func.__name__} at {datetime.now()} (Memory: {start_memory:.2f} MB)")
        
        # 执行函数
        # Execute function
        result = func(*args, **kwargs)
        
        # 记录结束时间和内存
        # Record end time and memory
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 计算差异
        # Calculate differences
        elapsed = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"{func.__name__} completed in {elapsed:.2f} seconds")
        print(f"Memory change: {memory_used:.2f} MB (Now: {end_memory:.2f} MB)")
        
        return result
    return wrapper


def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def print_section(title, char='=', width=80):
    """Print a section header with decorative characters"""
    print(char * width)
    print(f"{title.center(width)}")
    print(char * width)


def plot_feature_distribution(X, features=None, bins=20, figsize=(15, 10), use_gpu=True):
    """Plot feature distribution with GPU support"""
    if use_gpu:
        try:
            X_gpu = to_gpu(X)
            # GPU计算统计数据
            with gpu_context():
                if features is None:
                    if X_gpu.shape[1] > 10:
                        features = X_gpu.columns[:10]
                    else:
                        features = X_gpu.columns
                
                # 确定网格尺寸
                # Determine grid dimensions
                n_features = len(features)
                n_cols = min(3, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
                
                def _plot_feature(i, feature):
                    """Helper function for parallel plotting"""
                    if i < len(axes):
                        sns.histplot(X_gpu[feature], bins=bins, kde=True, ax=axes[i])
                        axes[i].set_title(f'Distribution of {feature}')
                        axes[i].set_xlabel(feature)
                        return True
                    return False
                
                # 并行处理绘图
                # Parallel processing for plotting
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(_plot_feature, i, feature): i 
                              for i, feature in enumerate(features)}
                    concurrent.futures.wait(futures)
                
                # 隐藏未使用的子图
                # Hide unused subplots
                for i in range(len(features), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.show()
                
                # 最后将结果转回CPU进行绘图
                X_gpu = to_cpu(X_gpu)
        except Exception as e:
            print(f"GPU处理失败，使用CPU: {e}")
            
    # 原有的绘图代码保持不变
    # ...


def create_confusion_matrix(y_true, y_pred, threshold=0.5, figsize=(8, 6)):
    """
    Create and visualize a confusion matrix with optimized calculations
    创建并可视化混淆矩阵，优化计算过程
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 转换概率预测为二进制
    # Convert probabilistic predictions to binary
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # 创建混淆矩阵
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 可视化混淆矩阵
    # Visualize confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])
    fig = plt.figure(figsize=figsize)
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.tight_layout()
    
    # 计算指标（使用更高效的方法）
    # Calculate metrics (using more efficient method)
    print(f"Classification Report (threshold={threshold}):")
    print(classification_report(y_true, y_pred_binary))
    
    # 不重复计算，直接从分类报告中获取数据
    # Don't recalculate; get data directly from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"详细指标 (threshold={threshold}):")
    print(f"  - 准确率 Accuracy: {accuracy:.4f}")
    print(f"  - 精确率 Precision: {precision:.4f}")
    print(f"  - 召回率 Recall: {recall:.4f}")
    print(f"  - F1分数 F1 Score: {f1:.4f}")
    
    plt.show()
    return cm, fig


def check_feature_importance(model, feature_names, top_n=20, plot=True):
    """
    Extract and optionally visualize feature importance from a model
    提取并可选地可视化模型的特征重要性
    
    Parameters:
    -----------
    model : 训练好的模型对象，必须具有feature_importances_属性
           Trained model object that must have feature_importances_ attribute
    feature_names : 特征名称列表
                   List of feature names
    top_n : 要显示的顶部特征数量
           Number of top features to display
    plot : 是否绘制重要性图
          Whether to plot importance graph
    
    Returns:
    --------
    importance_df : pandas.DataFrame
                   包含特征及其重要性的数据框
                   DataFrame containing features and their importance
    """
    if not hasattr(model, 'feature_importances_'):
        print("模型不提供特征重要性 / Model does not provide feature importance")
        return None
    
    # 创建特征重要性数据框
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # 使用并行处理进行排序（对于大型数据集）
    # Use parallel processing for sorting (for large datasets)
    def _process_importance():
        return importance_df.sort_values('Importance', ascending=False)
    
    # 对于小数据集不需要这种复杂处理，但展示并行处理技术
    # Not necessary for small datasets, but demonstrating parallel technique
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process_importance)
        importance_df = future.result()
    
    # 打印前N个特征
    # Print top N features
    print(f"前 {top_n} 个最重要特征 / Top {top_n} most important features:")
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.6f}")
    
    # 绘制特征重要性
    # Plot feature importance
    if plot:
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Feature Importance / 前 {top_n} 个特征重要性')
        
        # 添加数值标签以提高可读性
        # Add value labels for better readability
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + 0.002, 
                   p.get_y() + p.get_height()/2, 
                   f'{width:.4f}', 
                   ha='left', 
                   va='center')
            
        plt.tight_layout()
        plt.show()
    
    return importance_df


def parallelize_dataframe(df, func, n_cores=None):
    """
    Parallelize operations on pandas DataFrame
    并行化处理pandas数据框的操作
    
    Parameters:
    -----------
    df : pandas.DataFrame
        要处理的数据框
        DataFrame to process
    func : function
        应用于每个数据块的函数
        Function to apply to each chunk
    n_cores : int, optional
        要使用的核心数，默认为系统可用核心数
        Number of cores to use, defaults to available system cores
        
    Returns:
    --------
    pandas.DataFrame
        处理后的数据框
        Processed DataFrame
    """
    # 如果未指定核心数，使用全部可用核心数（减1以避免系统超载）
    # If cores not specified, use all available cores (minus 1 to avoid system overload)
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # 将数据拆分为块
    # Split data into chunks
    df_split = np.array_split(df, n_cores)
    
    # 创建进程池并将功能应用于每个块
    # Create pool and apply function to each chunk
    with multiprocessing.Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
        
    return df


def memory_usage_report(obj, name=None):
    """
    Report memory usage of a Python object
    报告Python对象的内存使用情况
    
    Parameters:
    -----------
    obj : object
        要分析的Python对象
        Python object to analyze
    name : str, optional
        对象的名称（用于报告）
        Name of object (for reporting)
        
    Returns:
    --------
    float
        对象使用的内存（MB）
        Memory used by object (MB)
    """
    # 转换为MB并打印
    # Convert to MB and print
    obj_name = name if name else type(obj).__name__
    memory = sys.getsizeof(obj) / 1024 / 1024
    
    print(f"内存使用 / Memory usage of {obj_name}: {memory:.3f} MB")
    
    # 对于pandas对象，使用其内置内存分析
    # For pandas objects, use their built-in memory analysis
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        detailed = obj.memory_usage(deep=True)
        if isinstance(detailed, pd.Series):
            print("详细内存使用 / Detailed memory usage (MB):")
            for key, val in detailed.items():
                print(f"  - {key}: {val/1024/1024:.3f} MB")
    
    return memory


def optimize_memory(dataframe):
    """优化DataFrame内存使用"""
    # 遍历所有列
    for col in dataframe.columns:
        # 对数值列进行降精度
        if dataframe[col].dtype == 'float64':
            dataframe[col] = dataframe[col].astype('float32')
        elif dataframe[col].dtype == 'int64':
            dataframe[col] = dataframe[col].astype('int32')
    return dataframe


def safe_concat(dataframes, axis=0, join='inner', ignore_index=True):
    """
    安全地连接数据框，避免空或全NA列引起的警告
    
    参数:
        dataframes (list): 要连接的DataFrame列表
        axis (int): 连接轴，0为行连接，1为列连接
        join (str): 连接方式，'inner'只保留共有列，'outer'保留所有列
        ignore_index (bool): 是否重置索引
        
    返回:
        pandas.DataFrame: 连接后的数据框
    """
    # 过滤空数据框
    non_empty_dfs = [df for df in dataframes if not df.empty]
    
    if not non_empty_dfs:
        return pd.DataFrame()  # 返回空数据框
    
    if len(non_empty_dfs) == 1:
        return non_empty_dfs[0].copy()  # 只有一个非空数据框时直接返回
    
    # 对行连接，确保列一致
    if axis == 0:
        # 找出共有列
        common_columns = set.intersection(*[set(df.columns) for df in non_empty_dfs])
        filtered_dfs = [df[list(common_columns)].dropna(axis=1, how='all') for df in non_empty_dfs]
    else:
        # 列连接不需要特殊处理
        filtered_dfs = non_empty_dfs
    
    # 安全连接
    return pd.concat(filtered_dfs, axis=axis, join=join, ignore_index=ignore_index)
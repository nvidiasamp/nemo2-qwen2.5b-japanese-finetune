# NCAA篮球锦标赛预测系统

[English](README_EN.md) | [中文](README_CN.md) | [日本語](README_JP.md)

## 简介

NCAA篮球锦标赛预测系统是一个先进的机器学习解决方案，旨在高精度预测NCAA篮球锦标赛比赛结果。该系统实现了一个复杂的预测流程，包括处理历史篮球数据、工程化相关特征、训练优化的XGBoost模型，以及为锦标赛对阵生成经过校准的胜率预测。

该系统专为[March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) Kaggle竞赛设计，该竞赛挑战参与者预测NCAA篮球锦标赛的比赛结果。

### 5.0版本主要改进

- **GPU加速**：通过cudf和cupy添加CUDA支持，在兼容硬件上显著提升性能
- **内存优化**：通过自适应批处理和精度降低增强内存管理
- **错误恢复能力**：改进了整个流程中的验证、优雅降级和错误恢复机制
- **扩展可视化**：全面的视觉分析，包括校准曲线和性别比较分析
- **多语言文档**：提供英文、中文和日文的完整文档

### 先前版本改进

- **双性别预测支持**：同时支持男子和女子NCAA篮球锦标赛
- **性能优化**：改进了并行处理和向量化操作，加快数据处理速度
- **内存效率**：为处理大型数据集提供更好的内存使用和缓存策略
- **健壮的错误处理**：整个流程中改进了验证和错误恢复能力

## 系统要求

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib
- tqdm
- psutil（用于内存监控）
- concurrent.futures（用于并行处理）
- cupy和cudf（可选，用于GPU加速）

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/ncaa-prediction-system.git
cd ncaa-prediction-system

# 创建虚拟环境（可选但推荐）
python -m venv myenv
source myenv/bin/activate  # Windows系统上：myenv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装GPU依赖（可选）
pip install cupy-cuda11x cudf-cuda11x
```

## 系统架构

系统遵循模块化架构设计，注重灵活性、可重复性和性能：

```
NCAA预测系统
├── 数据获取层
│   ├── 历史比赛数据加载
│   ├── 团队信息处理
│   └── 锦标赛结构分析
├── 特征工程层
│   ├── 团队表现统计
│   ├── 锦标赛晋级建模
│   ├── 对阵历史分析
│   └── 基于种子的特征生成
├── 模型训练层
│   ├── 性别特定模型训练
│   ├── 超参数优化
│   ├── 交叉验证框架
│   └── GPU加速学习
├── 预测与评估层
│   ├── 校准曲线分析
│   ├── Brier分数优化
│   ├── 预测分布分析
│   └── 风险调整策略
└── 可视化与报告层
    ├── 交互式性能图表
    ├── 性别比较分析
    ├── 特征重要性可视化
    └── 预测置信度分析
```

## 代码结构

项目分为多个模块，每个模块处理预测流程的特定方面：

- **main.py**：编排整个工作流程并提供命令行接口
- **data_preprocessing.py**：处理数据加载、探索和训练-验证集划分
- **feature_engineering.py**：从原始数据创建特征（团队统计、种子、对战记录）
- **train_model.py**：实现带有性别特定模型的XGBoost模型训练
- **submission.py**：生成锦标赛预测结果以供提交
- **evaluate.py**：包含评估指标和可视化工具
- **utils.py**：提供工具函数，包括GPU加速支持

## 使用方法

### 基本用法

```bash
python main.py --data_path ./data --output_path ./output --target_year 2025
```

### 高级选项

```bash
python main.py --data_path ./data \
               --output_path ./output \
               --train_start_year 2016 \
               --train_end_year 2024 \
               --target_year 2025 \
               --explore \
               --random_seed 42 \
               --n_cores 8 \
               --use_gpu \
               --generate_predictions
```

### 命令行参数

- `--data_path`：数据目录路径（默认：'../input'）
- `--output_path`：输出文件路径（默认：'../output'）
- `--explore`：启用数据探索和可视化（默认：False）
- `--train_start_year`：训练数据起始年份（默认：2016）
- `--train_end_year`：训练数据结束年份（默认：2024）
- `--target_year`：预测目标年份（默认：2025）
- `--random_seed`：随机种子，用于结果可重现（默认：42）
- `--n_cores`：并行处理使用的CPU核心数（默认：自动检测）
- `--use_cache`：使用缓存数据加速处理（默认：False）
- `--use_gpu`：为兼容的操作启用GPU加速（默认：False）
- `--xgb_trees`：XGBoost模型的树数量（默认：500）
- `--xgb_depth`：XGBoost模型的最大树深度（默认：6）
- `--xgb_lr`：XGBoost模型的学习率（默认：0.05）
- `--generate_predictions`：为所有可能的对阵生成预测（默认：False）
- `--output_file`：预测输出文件名（默认：基于时间戳）
- `--load_models`：加载预训练模型而不是训练新模型（默认：False）
- `--men_model`：男子模型文件路径（默认：None）
- `--women_model`：女子模型文件路径（默认：None）
- `--men_features`：男子特征文件路径（默认：None）
- `--women_features`：女子特征文件路径（默认：None）

## 数据要求

系统需要在数据目录中提供以下CSV文件：

- **MTeams.csv**：男子队伍信息
- **WTeams.csv**：女子队伍信息
- **MRegularSeasonCompactResults.csv**：男子常规赛结果
- **WRegularSeasonCompactResults.csv**：女子常规赛结果
- **MNCAATourneyCompactResults.csv**：男子锦标赛结果
- **WNCAATourneyCompactResults.csv**：女子锦标赛结果
- **MRegularSeasonDetailedResults.csv**：男子常规赛详细统计
- **WRegularSeasonDetailedResults.csv**：女子常规赛详细统计
- **MNCAATourneySeeds.csv**：男子锦标赛种子信息
- **WNCAATourneySeeds.csv**：女子锦标赛种子信息
- **SampleSubmissionStage1.csv**：样本提交格式

## 关键特性

### GPU加速

- 通过cupy和cudf库实现基于CUDA的加速
- 自适应GPU内存管理，带有回退机制
- 优化的张量操作，用于特征工程和模型训练
- 自动硬件检测，在不兼容时优雅降级到CPU

### 双性别预测

- 为男子和女子锦标赛分别训练单独的模型
- 针对每种锦标赛特点定制的性别特定特征工程
- 合并预测输出以提供全面的锦标赛覆盖
- 性别间预测模式的比较分析

### 高级特征工程

- 团队性能统计计算
- 种子信息处理
- 历史对战分析
- 锦标赛晋级概率估计
- 热门-冷门偏差校正
- 性别特定特征调整

### 性能优化

- 计算密集型操作的多核并行处理
- 兼容操作的GPU加速
- 内存缓存以避免重复计算
- 矢量化操作提高效率
- 内存使用监控和优化
- 用于性能跟踪的时间感知函数装饰器

### 健壮的评估

- 多种指标（Brier分数、对数损失、准确率、ROC AUC）
- 校准曲线分析
- 按性别划分的预测分布可视化
- 基于Brier分数特性的风险优化提交策略
- 男子和女子预测模型之间的比较分析

## 预测流程

1. **数据加载**：加载并预处理两种性别的历史篮球数据
2. **特征工程**：从原始数据创建预测特征，考虑性别特定因素
3. **模型训练**：为男子和女子锦标赛分别训练XGBoost模型
4. **评估**：使用多种指标评估模型性能
5. **预测生成**：为所有可能的锦标赛对阵创建预测
6. **风险策略应用**：为Brier分数应用最优风险策略
7. **提交创建**：格式化预测以供竞赛提交

## 理论洞察

系统实现了几个理论洞察以提高预测准确性：

- **Brier分数优化**：对于胜率约为33.3%的预测，应用策略性风险调整以优化预期Brier分数。
- **热门-冷门偏差校正**：系统校正了对强队（低种子）的系统性低估和对弱队（高种子）的高估。
- **时间感知验证**：使用较新赛季进行验证，以更好地反映篮球预测的时间性质。
- **性别特定建模**：单独的模型捕捉男子和女子篮球锦标赛的独特特征。
- **校准理论**：实现概率校准技术，确保预测的概率准确反映真实的获胜可能性。

## 示例结果

系统生成几个输出文件：

- 为男子和女子锦标赛训练的模型文件（men_model.pkl, women_model.pkl）
- 特征缓存文件（men_features.pkl, women_features.pkl）
- 预测提交文件（submission_YYYYMMDD_HHMMSS.csv）
- 模型评估指标和可视化结果
- 男子和女子预测之间的比较分析

## 高级用法

### GPU加速

```python
from utils import gpu_context, to_gpu, to_cpu

# 检查GPU是否可用
with gpu_context(use_gpu=True) as gpu_available:
    if gpu_available:
        print("GPU加速已启用")
        # 将数据移至GPU
        X_gpu = to_gpu(X_train)
        y_gpu = to_gpu(y_train)
        
        # 在GPU上处理
        # ... 处理步骤 ...
        
        # 将结果移回CPU
        X_processed = to_cpu(X_gpu)
        y_processed = to_cpu(y_gpu)
    else:
        print("GPU不可用，使用CPU")
        X_processed = X_train
        y_processed = y_train
```

### 训练性别特定模型

```python
from train_model import train_gender_specific_models
from utils import save_features

# 为两种性别准备特征
m_features, m_targets = merge_features(m_train_data, m_team_stats, m_seed_features, m_matchup_history)
w_features, w_targets = merge_features(w_train_data, w_team_stats, w_seed_features, w_matchup_history)

# 训练性别特定模型
models = train_gender_specific_models(
    m_features, m_targets, w_features, w_targets,
    m_tourney_train, w_tourney_train,
    random_seed=42, save_models_dir='./models'
)

# 访问单独的模型
men_model = models['men']['model']
women_model = models['women']['model']
```

### 生成组合预测

```python
from submission import prepare_all_predictions, create_submission

# 为两种性别生成预测
all_predictions = prepare_all_predictions(
    model, features_dict, data_dict, 
    model_columns=model_columns,
    year=2025, 
    gender='both'  # 处理男子和女子的对阵
)

# 创建提交文件
submission = create_submission(all_predictions, sample_submission, 'submission_2025.csv')
```

## 性能注意事项

- 特征工程是流程中最耗时的部分；使用`--use_cache`标志可重用先前计算的特征。
- GPU加速显著提高性能，但需要兼容的硬件和驱动程序。
- 对于极大型数据集，调整`n_cores`参数以平衡速度和内存使用。
- 系统包含自动批次大小优化，以有效管理内存使用。

## 可视化

系统生成几个可视化结果以帮助理解模型性能：

- 男子和女子锦标赛的预测分布图
- 显示预测与实际胜率的校准曲线
- 突出显示最具预测性因素的特征重要性图
- 显示男子和女子预测差异的比较图
- 内存和性能分析图表

## 参考资料

- March Machine Learning Mania 2025：[https://www.kaggle.com/competitions/march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
- XGBoost：[https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Brier分数：[https://en.wikipedia.org/wiki/Brier_score](https://en.wikipedia.org/wiki/Brier_score)
- NCAA锦标赛：[https://www.ncaa.com/march-madness](https://www.ncaa.com/march-madness)
- RAPIDS cuDF：[https://docs.rapids.ai/api/cudf/stable/](https://docs.rapids.ai/api/cudf/stable/)
- CuPy：[https://cupy.dev/](https://cupy.dev/)

## 作者

赵俊茗 (Junming Zhao)

## 许可证

MIT许可证

---

本README提供了NCAA篮球锦标赛预测系统的全面概述，包括设置说明、使用示例和关键技术细节。如有问题或贡献，请在仓库中提出issue。
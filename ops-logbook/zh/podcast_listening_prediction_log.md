# Kaggle竞赛：预测播客收听时长 - 操作日志

## 项目概述
- **竞赛名称**：Predict Podcast Listening Time (预测播客收听时长)
- **目标**：根据提供的用户和播客相关特征，预测用户收听特定播客单集的时长
- **评估指标**：均方根误差 (RMSE)
- **数据位置**：`/home/cho/workspace/kaggle/input/playground-series-s5e4`

## 日志记录

### 初始化阶段 - [2025/4/27]

#### 1. 项目设置
**操作**：
- 创建项目基本结构
- 设置数据读取与处理脚本
- 创建初步的EDA (探索性数据分析) 脚本
- 在 `/home/cho/workspace/kaggle/Predict-Podcast-Listening-Time` 中创建 `src` 目录
- 创建 `src/utils.py`: 包含数据加载和路径管理等辅助函数。
- 创建 `src/data_exploration.py`: 用于加载数据并执行初步的数据探索性分析（EDA）。
- 创建 `src/feature_engineering.py`: 包含数据预处理、特征生成、编码和缩放的函数框架。
- 创建 `src/model_training.py`: 包含模型训练、评估和保存的函数框架。
- 创建 `src/predict.py`: 包含加载模型、处理测试数据并生成提交文件的函数框架。
- 创建 `requirements.txt`: 包含项目所需的Python依赖包列表 (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`)。
- 创建Python虚拟环境 (`.venv`): 使用 `python3 -m venv .venv` 在项目根目录创建，用于隔离项目依赖。

**虚拟环境使用说明**:
  - **激活环境 (Bash/Zsh)**: `source .venv/bin/activate`
  - **安装依赖**: (激活环境后) `pip install -r requirements.txt`
  - **退出环境**: `deactivate`

**遇到的问题**：
- 暂无

**解决方案**：
- 暂无

#### 2. 数据初探
**操作**：
- 加载训练和测试数据集
- 检查数据基本信息（行数、列数、数据类型、缺失值等）
- 分析目标变量 `Listening_Time_minutes` 的分布

**发现**：
- 待填写数据探索结果

**下一步计划**：
- 进行更详细的EDA
- 开始特征工程
- 尝试基线模型 

### 代码可复现性与环境管理 (基于书籍学习) - [2025/4/27]

**学习要点**:
- **`requirements.txt`**: 定义项目所需的 Python 库及其精确版本，是保证环境一致性的关键。使用 `pip freeze > requirements.txt` 生成和更新。
- **Docker**: 提供更深层次的环境隔离和打包。通过 `Dockerfile` 定义镜像构建步骤，将代码、依赖、甚至操作系统层打包成镜像 (Image)，然后运行为容器 (Container)。
    - `docker build`: 构建镜像。
    - `docker run`: 运行容器。
- **模型服务**: 使用 Flask/FastAPI 等框架将模型包装成 API，用于实时预测。通常需要 WSGI/ASGI 服务器 (如 Gunicorn/Uvicorn) 在生产环境运行。

**对本竞赛的意义**:
- **`requirements.txt` (高相关性)**: **必须维护**。确保训练和预测环境的一致性，方便协作和后续运行。随着新库的引入，应定期更新此文件。
- **Docker (中等相关性)**: **可选**。对于当前竞赛可能不是必需的，但如果项目流程变得非常复杂或需要分享包含特定系统依赖的完整环境时，Docker 是一个强大的工具。了解其概念有益于更广泛的软件工程实践。
- **模型服务 (低相关性)**: 对于生成提交文件的 Kaggle 竞赛目标，这部分内容**不直接适用**。

**下一步计划**:
- 在安装新库后，**立即更新** `requirements.txt` 文件。
- （可选）如果未来流程复杂度增加，考虑使用 Docker 打包训练/预测流程。

### 代码结构重构与配置管理 (依据书本原则) - [记录日期]

**行动**: 基于 "Approaching (almost) any machine learning problem" 的指导，开始重构 `src` 目录下的代码，以实现更健壮的机器学习流程。

**关键变更**: 
1.  **创建 `src/config.py`**: 引入中央配置文件来管理路径、文件名、超参数、特征列表等常量，提高代码的可读性和可维护性。
2.  **后续步骤**: 将修改 `utils.py`, `data_exploration.py`, `feature_engineering.py`, `model_training.py`, 和 `predict.py` 以使用 `config.py` 中的常量，并实现正确的交叉验证流程和特征工程（分离拟合与转换）。

**目标**: 提升代码质量、可复现性，并为后续的特征工程、模型训练和评估打下坚实基础。

### 更新 `utils.py` - [记录日期]

**行动**: 修改了 `src/utils.py` 文件。

**关键变更**:
- 移除了硬编码路径的 `get_data_path` 函数。
- 导入了 `src/config.py` 并更新 `load_data` 以使用配置中的路径，并添加了文件存在性检查。
- 添加了 `save_dataframe` 和 `load_dataframe` 函数，用于高效地读写 Parquet 格式的数据（例如处理后的特征）。
- 添加了 `save_object` 和 `load_object` 函数，使用 `joblib` 来保存/加载 Python 对象（例如训练好的模型、编码器、缩放器）。
- 添加了 `setup_logger` 函数，用于设置日志记录，将信息同时输出到控制台和 `output/logs` 目录下的日志文件。

**注意**: 使用 Parquet 格式需要 `pyarrow` 库。需要确保它已添加到 `requirements.txt` 并安装。 

### 更新 `data_exploration.py` - [2025/4/27]

**行动**: 修改了 `src/data_exploration.py` 文件。

**关键变更**:
- 导入了 `config` 和 `utils.setup_logger`，并初始化了针对此脚本的日志记录器 (`data_exploration.log`)。
- 使用 `logging` 替代了 `print` 语句，以便更好地跟踪执行过程。
- 修改了 `explore_data` 函数，增加了 `df_name` 和 `save_plots` 参数。
- 更新了绘图逻辑，现在可以根据 `save_plots` 参数将目标变量分布图保存到 `config.PLOT_DIR` 中，而不是仅在屏幕上显示。
- 更新了 `if __name__ == "__main__"` 块，使用 `config` 中的文件路径和新的 `utils.load_data` 函数加载数据，并调用 `explore_data` (示例中开启了训练数据绘图保存)。 

### 更新 `feature_engineering.py` - [2025/4/27]

**行动**: 重构了 `src/feature_engineering.py` 文件，以分离拟合（fitting）和应用（transforming）特征转换的步骤。

**关键变更**:
- 导入了 `config`, `logging`, `utils` 以及 `sklearn.preprocessing` 中的 `StandardScaler`, `OneHotEncoder`, `SimpleImputer`。
- 设置了针对此脚本的日志记录器 (`feature_engineering.log`)。
- 更新了 `preprocess_data` 函数，使用 `SimpleImputer` 处理数值和分类特征的缺失值（作为示例，具体策略需基于EDA调整），并依赖 `config` 中的特征列表。
- 更新了 `generate_features` 函数（目前主要包含日志记录和待办事项）。
- **创建了 `fit_transformers` 函数**: 
    - 接收训练数据。
    - 拟合 `StandardScaler` 到 `config.NUMERICAL_FEATURES`。
    - 拟合 `OneHotEncoder` (设置 `handle_unknown='ignore'`) 到 `config.CATEGORICAL_FEATURES`。
    - 使用 `utils.save_object` 将拟合好的 `scaler` 和 `encoder` 分别保存到 `config.SCALER_FILE` 和 `config.ENCODER_FILE`。
- **创建了 `apply_transformations` 函数**:
    - 接收数据帧以及已加载的 `scaler` 和 `encoder`。
    - 应用 `scaler.transform` 到数值特征。
    - 应用 `encoder.transform` 到分类特征，处理生成的特征名称，并与原数据合并，删除原始分类列。
- 移除了旧的 `encode_categorical_features`, `scale_numerical_features`, 和 `apply_feature_engineering` 函数及示例用法。

**重要**: 此重构使得特征工程步骤能在交叉验证的每一折内正确执行：在训练折叠上调用 `fit_transformers`，然后在训练和验证折叠上调用 `apply_transformations`（使用同一组拟合好的转换器），有效防止数据泄漏。 
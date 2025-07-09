# 复现 NVIDIA NIM 部署微调模型 - 过程记录与总结

## 1. 引言

本文档旨在记录复现 NVIDIA 官方博客文章 "[Deploying Fine-tuned AI Models with NVIDIA NIM](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)" 的全过程。目标是理解并实践如何使用 NVIDIA NIM (NVIDIA Inference Microservices) 部署经过微调的 AI 模型。

本文还将详细记录在本地环境复现时与博客所述环境的差异、所做的相应调整、具体操作步骤、遇到的问题及其解决方案。最终，本文档将作为技术分享和会议发表的基础材料。

**项目遵循的 Cursor Rule:** `reproduce_nvidia_nim_blog.mdc`

## 2. 博客内容核心解读

NVIDIA的博客文章 "[Deploying Fine-tuned AI Models with NVIDIA NIM](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)" (后文简称"博客") 详细阐述了如何利用 **NVIDIA NIM (NVIDIA Inference Microservices)** 来高效部署经过微调（Fine-tuned）的AI大语言模型（LLM）。

**博客的核心目的与NIM要解决的问题：**

博客的核心主旨是展示NIM如何**简化和加速微调LLM的部署过程，特别是那些直接修改了模型权重（如通过SFT）的模型**。传统上，将这些微调后的模型投入生产并获得最佳推理性能，通常涉及到复杂且耗时的手动优化步骤，例如选择合适的推理引擎、针对特定硬件进行编译优化、以及配置服务接口等。NIM旨在通过以下方式解决这些痛点：

1.  **自动化优化与部署：** NIM能够在单步部署流程中，为用户提供的模型权重（无论是基础模型还是SFT微调模型）自动在本地构建和加载一个针对特定GPU硬件优化的TensorRT-LLM推理引擎。这意味着用户无需深入了解TensorRT-LLM的底层细节即可获得高性能推理。
2.  **标准化与易用性：** NIM将复杂的推理后端封装在Docker容器中，并提供与OpenAI API兼容的标准化接口。这使得开发者可以轻松地将NIM部署的模型集成到现有的AI应用或工作流中。
3.  **灵活性与高性能：** 支持多种微调策略和模型，并通过深度集成TensorRT-LLM来确保模型以高效率（低延迟、高吞吐量）运行。

**关键技术组件解读：**

*   **NVIDIA NIM (NVIDIA Inference Microservices):**
    *   **是什么：** 一套预构建的、针对NVIDIA GPU优化的推理微服务。它们以Docker容器的形式提供，封装了运行特定AI模型（如LLM）所需的一切。
    *   **为什么用：** 主要为了简化部署。用户选择一个与其模型基础架构相匹配的NIM镜像，NIM负责处理后续的优化和运行。
    *   **核心功能：** 对于LLM，NIM的核心功能之一是集成了TensorRT-LLM。当NIM服务启动并指向用户提供的模型权重（通过 `NIM_FT_MODEL` 参数）时，它会利用TensorRT-LLM**动态构建**（或加载之前缓存的）一个针对当前模型和GPU硬件高度优化的推理引擎。这个"动态构建"的能力对于SFT微调模型至关重要，因为它能确保推理引擎与新的、微调后的权重完全匹配。

*   **TensorRT-LLM:**
    *   **是什么：** NVIDIA的开源库，专门用于在NVIDIA GPU上加速LLM的推理。它能将标准的LLM（如来自Hugging Face的模型）转换为针对特定NVIDIA GPU硬件高度优化的格式。
    *   **为什么用：** 实现极致的推理性能，包括更低的延迟和更高的吞吐量。
    *   **如何工作：** 它应用了多种优化技术，如算子融合、量化（如FP8、INT8）、张量并行、流水线并行等。

*   **SFT (Supervised Fine-tuning) 与模型权重：**
    *   **是什么：** 一种微调技术，通过在特定任务的标记数据集上训练，直接修改预训练LLM的基础权重，使其更适应特定领域或任务。
    *   **博客关注点：** 博客特别强调了NIM对SFT模型的支持。因为SFT直接改变了权重，所以推理引擎需要能够适应这些新权重以保证最佳性能，这正是NIM通过动态构建TensorRT-LLM引擎所实现的。

*   **Docker:**
    *   **是什么：** 一种容器化技术。
    *   **为什么用：** NIM服务通过Docker容器打包和运行，极大地简化了环境配置、依赖管理和部署的可移植性。用户无需担心本地环境与NIM运行所需的复杂依赖（如特定版本的CUDA、cuDNN、TensorRT-LLM等）之间的冲突。

**博客描述的核心流程概览 (以SFT模型为例)：**

1.  **前提条件：** 准备好NVIDIA GPU环境、`git-lfs`、NGC API Key。
2.  **环境设置：** 导出必要的环境变量，如 `NGC_API_KEY` 和 `NIM_CACHE_PATH` (用于缓存优化引擎)。
3.  **模型准备：**
    *   获取SFT微调模型的权重（博客示例为 `nvidia/OpenMath2-Llama3.1-8B`）。
    *   **关键点：** 所用微调模型的基础模型架构必须是NVIDIA NIM支持的。例如，如果你的SFT模型是基于Llama-3.1-8B的，那么你需要使用NIM中针对Llama-3.1-8B基础模型的镜像。
    *   设置 `MODEL_WEIGHT_PARENT_DIRECTORY` 指向存放模型权重的父目录。
4.  **选择NIM容器镜像 (`IMG_NAME`):**
    *   根据你模型的基础架构，从NVIDIA API Catalog (NGC) 中选择对应的NIM for LLMs容器镜像。例如，博客中 `OpenMath2-Llama3.1-8B` (一个SFT模型) 使用的是 `nvcr.io/nim/meta/llama-3.1-8b-instruct:1.3.0` (一个针对其基础模型Llama-3.1-8B Instruct的NIM镜像)。
5.  **（可选）选择性能配置文件 (`NIM_MODEL_PROFILE`):**
    *   NIM允许通过此环境变量指定一个预定义的性能优化配置文件，例如侧重低延迟 (`latency`) 或高吞吐量 (`throughput`)。这些配置文件通常包含了针对特定硬件（如H100）、精度（如BF16、FP8）、并行策略（如TP、PP）的优化参数。
    *   博客中的 "カスタム パフォーマンス プロファイルで最適化された TensorRT-LLM エンジンのビルド" (使用自定义性能配置文件构建优化的TensorRT-LLM引擎) 部分就详细讨论了这一点。
    *   **如果未指定或没有直接兼容的预构建profile**：NIM会尝试自动检测硬件和模型，并动态构建一个优化的TensorRT-LLM引擎。它会选择一个通用的、可构建的配置（如我们实验中观察到的 `tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`）。
6.  **启动NIM微服务 (通过 `docker run`):**
    *   命令中包含GPU指定、API密钥、模型权重路径 (通过 `-v` 挂载到容器内的 `NIM_FT_MODEL` 所指向的路径)、服务暴露的模型名 (`NIM_SERVED_MODEL_NAME`)、可选的性能配置 (`NIM_MODEL_PROFILE`)，以及模型权重和缓存的路径挂载。
    *   **核心机制**：当Docker容器启动时，NIM服务会读取 `NIM_FT_MODEL` 指向的权重，并结合（可选的）`NIM_MODEL_PROFILE` 或其自动选择的构建策略，利用TensorRT-LLM来构建（或从缓存加载）一个针对当前GPU优化的推理引擎。这个过程可能需要几分钟。
7.  **与模型交互：** 通过标准的OpenAI API兼容接口 (通常是 `http://localhost:8000/v1`) 与部署的模型进行通信。

**为什么博客中看似不同的部分在您实践中结果相似？**

您提到博客中的 "SFT モデルを使用した例" (使用SFT模型的示例) 和 "カスタム パフォーマンス プロファイルで最適化された TensorRT-LLM エンジンのビルド" (使用自定义性能配置文件构建) 部分，在您的复现过程中可能得到了相似的执行结果。这是因为：

*   **两者都依赖NIM的核心部署机制：** 无论是直接部署一个SFT模型（如博客第一部分），还是在部署时尝试指定一个自定义性能配置文件（如博客第二部分），**底层的NIM工作流程是相似的**。NIM都会获取模型权重，然后尝试构建或加载一个优化的TensorRT-LLM引擎。
*   **`NIM_MODEL_PROFILE` 的作用：**
    *   如果提供了**有效且兼容的** `NIM_MODEL_PROFILE`，NIM会按照该配置文件的指示进行优化。
    *   如果您提供的 `NIM_MODEL_PROFILE` **无效、不兼容，或者您根本没有提供这个参数**（就像我们后来在实验中确认的那样，因为没有找到直接适用于RTX 6000 Ada + Llama-3.2-1B的预构建profile），NIM会**回退到其自动优化行为**。它会选择一个通用的、可构建的策略（如 `tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`）来动态构建引擎。
*   **您的实践情况：** 在我们的复现中，`list-model-profiles` 命令显示，对于您使用的NIM镜像和RTX 6000 Ada GPU，并没有直接兼容的预构建性能配置文件。因此，无论您是否尝试在Docker命令中加入 `-e NIM_MODEL_PROFILE=...` （除非恰好指定了一个NIM内部能识别并间接使用的构建参数），NIM最终的行为都会是相似的：它会进行自动的、基于"buildable"策略的引擎构建。

因此，博客的这两个部分更多的是展示了NIM部署流程的**不同配置选项和观察侧面**：
*   第一个主要部分展示了**基本SFT模型的部署流程**，强调NIM能够处理自定义权重。
*   第二个主要部分则深入探讨了**如何通过性能配置文件进行更细致的优化控制**，并展示了如何查询可用的profiles。

在没有可用预构建profile的情况下，这两个流程最终都会触发NIM的动态引擎构建机制，从而导致相似的日志输出（如引擎构建过程）和最终的服务行为。

**对本地复现的关键启示：**
*   **GPU显存是重要约束：** 本地GPU显存（约48GB）远小于博客示例（80GB），因此必须选择参数量较小的模型（如 `meta-llama/Llama-3.2-1B`）进行复现。
*   **`IMG_NAME` 的准确性至关重要：** 必须从NVIDIA API Catalog为模型的基础架构找到正确的NIM容器镜像名称。
*   **路径映射需正确：** Docker的卷挂载（`-v` 参数）必须正确设置，以确保模型权重对容器内部可见。
*   **理解NIM的自动化能力：** 即使没有完美的预构建性能配置文件，NIM依然能够通过动态构建TensorRT-LLM引擎来提供优化服务。

**本次复现说明更新：** 鉴于本地环境显存限制（约48GB，NVIDIA RTX 6000 Ada），本次复现选用 `meta-llama/Llama-3.2-1B` 作为基础模型进行NIM部署实践。我们重点观察了NIM在没有显式指定兼容的自定义性能配置文件时，其自动检测硬件、选择通用构建策略（如`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`）并动态为本地模型权重构建优化TensorRT-LLM引擎的过程和结果。

## 3. 本地复现环境

### 3.1 博客描述环境

*   GPU: 80 GB GPU 内存 (具体型号未指明，可能是 A100 或 H100)
*   软件: `git-lfs`
*   其他: NGC API Key

### 3.2 我的本地环境

*   **操作系统:** Ubuntu 20.04.6 LTS (focal)
*   **GPU型号与数量 (本项目使用):** 
    *   GPU 0: NVIDIA RTX 6000 Ada Generation (本项目指定使用此GPU)
    *   (GPU 1: NVIDIA T600 - 系统检测到，但本项目不使用)
*   **显存大小 (GPU 0):** 49140MiB (约 48GB)
*   **NVIDIA 驱动版本:** 535.183.01
*   **Docker 版本:** Docker version 28.0.4, build b8034c0
*   **CUDA 版本 (驱动支持):** 12.2 (注: `nvcc` 命令未找到，可能未单独安装 CUDA Toolkit 或不在 PATH 中)
*   **git-lfs 版本:** git-lfs/3.6.1 (GitHub; linux amd64; go 1.23.3)
*   **主要差异点:** 
    *   使用的GPU (NVIDIA RTX 6000 Ada Generation) 最大单卡显存约为 48GB，远小于博客建议的 80GB。
    *   因此，**模型选择调整为 `meta-llama/Llama-3.2-1B`** (或其SFT微调版本，如果适用且能找到)。这可能会严重影响可运行的模型大小和推理性能，需要特别注意模型选择和资源配置。

## 4. 复现步骤与操作记录

*(按照 `reproduce_nvidia_nim_blog.mdc` 规则中的核心步骤以及博客流程，详细记录每一步)*

### 4.1 环境准备

*   **NGC API Key 设置:**
    *   实际操作：已在终端执行以下命令设置 `NGC_API_KEY` 环境变量。为安全起见，API Key的具体值已在此处隐去，请确保在您的实际操作环境中正确设置。
    ```bash
    export NGC_API_KEY="nvapi-02nWXga...JJ5Ix" # 此处为示意，实际已设置
    ```
*   **NIM Cache Path 设置:**
    *   实际操作：已在终端执行以下命令设置 `NIM_CACHE_PATH`，创建对应目录并设置权限。
    ```bash
    export NIM_CACHE_PATH=/tmp/nim/.cache 
    mkdir -p $NIM_CACHE_PATH
    chmod -R 777 $NIM_CACHE_PATH
    ```
    *   注：`chmod -R 777` 提供了非常宽松的权限，适用于开发测试阶段以避免权限问题。在生产环境中应考虑更安全的权限设置。

### 4.2 模型获取

*   **`git lfs` 初始化与 `models` 目录创建:**
    *   为确保能正确处理大型模型文件，首先执行了 `git lfs install` (或确认已全局初始化)。
    *   然后，在项目根目录 (`/home/cho/workspace/MTG_ambassador/finetuning_model_deploy`)下创建了 `models` 子目录用于存放下载的模型。
    ```bash
    # git lfs install # (建议执行)
    mkdir -p models
    ```
*   **Hugging Face CLI 登录与模型克隆:**
    *   尝试克隆 `meta-llama/Llama-3.2-1B` 时遇到认证失败。通过以下步骤解决：
        1.  在 Hugging Face 网站 ([https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)) 确认已请求并获得模型访问权限。
        2.  访问 Hugging Face Access Tokens 设置页面 ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))，生成/更新了一个具有访问受门控仓库 (gated repositories) 权限的访问令牌。
        3.  使用新令牌重新执行 `huggingface-cli login` 并将其添加为Git凭证。
            ```bash
            # huggingface-cli login # (粘贴有权限的令牌并确认为Git凭证)
            ```
    *   成功登录后，执行以下命令克隆 `meta-llama/Llama-3.2-1B` 模型到 `models/Llama-3.2-1B` 目录：
        ```bash
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B models/Llama-3.2-1B
        ```
        模型已成功下载到 `/home/cho/workspace/MTG_ambassador/finetuning_model_deploy/models/Llama-3.2-1B`。

*   **设置 `MODEL_WEIGHT_PARENT_DIRECTORY` 环境变量:**
    *   此环境变量指向包含所有模型权重文件夹的父目录。对于本项目，即为 `models` 目录。
    *   在项目根目录下执行 (或已执行) 以下命令进行设置：
    ```bash
    export MODEL_WEIGHT_PARENT_DIRECTORY=${PWD}/models
    # ${PWD} 在此上下文中为 /home/cho/workspace/MTG_ambassador/finetuning_model_deploy
    # 因此 MODEL_WEIGHT_PARENT_DIRECTORY 被设置为 /home/cho/workspace/MTG_ambassador/finetuning_model_deploy/models
    ```
*   **关键后续步骤:** 需要在 NVIDIA API Catalog 中查找 `meta-llama/Llama-3.2-1B` (或其基础模型，如果适用) 对应的NIM基础模型镜像 (`IMG_NAME`)。

### 4.3 NIM 微服务启动

**背景说明：为何在已有模型权重后仍需NIM模型镜像 (`$IMG_NAME`)？**

我们已经下载了 `meta-llama/Llama-3.2-1B` 的模型权重（例如 `.safetensors` 文件）。这些权重是模型的"知识核心"。然而，要让这些权重能够高效地对外提供推理服务，我们还需要一个专门的运行环境和推理引擎。这正是 NVIDIA NIM 模型镜像 (`$IMG_NAME`) 所提供的。

NIM 模型镜像是一个预配置的、高度优化的 Docker 镜像，它包含了：
1.  **推理服务器软件：** 如 NVIDIA Triton Inference Server，用于管理模型生命周期、处理请求和响应。
2.  **TensorRT-LLM 引擎/库：** NVIDIA 的核心技术，用于将大语言模型转换为针对特定NVIDIA GPU硬件高度优化的格式，以实现最佳推理性能（低延迟、高吞吐量）。
3.  **模型优化和构建逻辑：** 当NIM服务启动并指向我们提供的模型权重时（通过 `NIM_FT_MODEL` 参数），它会利用TensorRT-LLM动态构建（或加载缓存的）一个针对当前模型和GPU硬件优化的推理引擎。这个过程对于SFT微调模型尤其重要，因为它能确保推理引擎与新的权重完全匹配。
4.  **标准化的API接口：** 通常提供与OpenAI API兼容的接口，方便应用程序集成。
5.  **所有必要的依赖：** 包括CUDA驱动、库文件等，确保模型在NVIDIA GPU上顺利运行。

简单来说，模型权重是"食材"，而NIM模型镜像是"设备齐全的专业厨房+能将食材变为成品并提供服务的厨师团队"。它将静态的权重转化为一个动态的、高效的、可生产使用的推理服务。虽然理论上可以手动搭建这一切，但NIM极大地简化了部署和优化的复杂性。

*   **获取 NIM 基础模型镜像 (`IMG_NAME`):**
    *   根据NVIDIA API Catalog (NGC) 的查询结果，针对 `meta-llama/Llama-3.2-1B` 模型（特别是其指令微调版本的基础），应使用的NIM镜像为 `nvcr.io/nim/meta/llama-3.2-1b-instruct:1`。
    *   实际操作：将在终端设置此环境变量。
    ```bash
    export IMG_NAME="nvcr.io/nim/meta/llama-3.2-1b-instruct:1"
    ```
*   **关于性能配置文件 (`NIM_MODEL_PROFILE`):**
    *   NIM在为模型构建优化的TensorRT-LLM推理引擎时，允许通过性能配置文件来指导其优化侧重点。博客中提到了两种主要类型：
        1.  **延迟 (Latency):** 优先减少单个推理请求的处理时间，适用于对实时响应敏感的应用（如聊天机器人）。
        2.  **吞吐量 (Throughput):** 优先最大化单位时间内处理的请求数量，适用于需要处理大量并发或批处理任务的场景。
    *   **选择方式：**
        *   **自动选择：** 如果不指定，NIM会尝试根据模型和硬件自动选择一个合适的配置文件。
        *   **手动指定：** 可以通过在 `docker run` 命令中使用 `-e NIM_MODEL_PROFILE=<profile_name>` 参数来显式指定。例如，博客中使用了 `tensorrt_llm-h100-bf16-tp2-pp1-latency`。具体的profile名称通常包含了硬件、精度、并行策略和优化目标等信息。
    *   **建议：** 初步运行时可以不指定，观察NIM的默认性能。如需进一步优化，可查阅NVIDIA NIM官方文档，寻找适用于您GPU (RTX 6000 Ada) 和模型 (Llama-3.2-1B) 的推荐性能配置文件，并在 `docker run` 命令中指定。

*   **使用脚本启动NIM服务 (推荐方法):**

    为了简化环境变量设置和Docker命令的执行，项目在 `scripts/` 目录下提供了两个脚本：

    1.  **`scripts/01_set_env_vars.sh`**: 
        *   **作用**: 此脚本负责设置所有启动NIM服务所需的环境变量，包括 `NGC_API_KEY`, `NIM_CACHE_PATH`, `MODEL_WEIGHT_PARENT_DIRECTORY`, 和 `IMG_NAME`。
        *   **重要**: 在首次使用前，您**必须**编辑此文件，将占位符 `YOUR_NGC_API_KEY_HERE` 替换为您真实的NGC API密钥。
        *   **使用**: 在终端中，导航到项目根目录 (`/home/cho/workspace/MTG_ambassador/finetuning_model_deploy`) 并执行以下命令将其加载到当前shell会话：
            ```bash
            source ./scripts/01_set_env_vars.sh
            ```
        *   脚本会自动检测项目路径来设置 `MODEL_WEIGHT_PARENT_DIRECTORY`。

    2.  **`scripts/03_run_nim_docker.sh`**: 
        *   **作用**: 此脚本包含实际的 `docker run` 命令，用于启动NIM服务。它会使用已由 `01_set_env_vars.sh` 加载的环境变量，并根据本地 `Llama-3.2-1B` 模型进行了参数配置（如模型路径、服务名称、GPU指定为 `device=0`、共享内存大小等）。
        *   **前置条件**: 必须先成功 `source` 了 `01_set_env_vars.sh` 脚本。
        *   **使用**: 在环境变量加载后，于项目根目录下执行：
            ```bash
            ./scripts/03_run_nim_docker.sh
            ```
        *   此脚本会先检查所需环境变量是否已设置，然后尝试创建并设置 `NIM_CACHE_PATH` 的权限，最后执行 `docker run` 命令。
        *   默认情况下，性能配置文件 (`NIM_MODEL_PROFILE`) 由NIM自动选择，共享内存 (`--shm-size`) 设置为16GB。您可以根据需要直接修改此脚本中的这些值。

*   **运行 Docker 命令 (手动参考):**
    *   以下命令是 `scripts/03_run_nim_docker.sh` 脚本中执行的核心命令。脚本化执行是推荐的方式，但此处保留手动命令作为参考，并已根据本地 `Llama-3.2-1B` 模型和环境进行了调整。
    *   说明：由于我们下载的是 `meta-llama/Llama-3.2-1B` (基础版) 的权重，而使用的NIM镜像是针对 `instruct` 版本。NIM会尝试为提供的权重构建优化引擎。`NIM_FT_MODEL` 应指向基础版权重的文件夹。
    ```bash
    # docker run --rm --gpus '"device=0"' \
    #     --user $(id -u):$(id -g) \
    #     --network=host \
    #     --shm-size=16GB \ # (1B模型可能不需要32GB，根据实际情况调整)
    #     -e NGC_API_KEY \
    #     -e NIM_FT_MODEL=/opt/weights/hf/Llama-3.2-1B \ # 假设模型文件夹名为 Llama-3.2-1B
    #     -e NIM_SERVED_MODEL_NAME=Llama-3.2-1B-custom \ # 自定义服务名，启动后需确认实际查询名
    #     -e NIM_MODEL_PROFILE=tensorrt_llm-h100-bf16-tp2-pp1-latency \ # (可能需要根据实际GPU和模型大小调整profile)
    #     -v $NIM_CACHE_PATH:/opt/nim/.cache \
    #     -v $MODEL_WEIGHT_PARENT_DIRECTORY:/opt/weights/hf \
    #     $IMG_NAME
    # (记录实际执行的完整命令)
    ```

#### 4.3.1 故障排查：端口冲突 (Port 8000 in use)

在首次尝试运行 `./scripts/03_run_nim_docker.sh` 后，遇到了以下错误：
```
ERROR 2025-05-22 07:34:00.333 server.py:170] [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```
这表明主机上的端口8000已经被其他进程占用，NIM服务无法启动。

**解决步骤：**

1.  **识别占用端口的进程**：
    使用以下命令之一查找占用端口8000的进程：
    ```bash
    sudo netstat -tulnp | grep :8000
    # 或
    sudo ss -tulnp | grep :8000
    ```
    在本次排查中，输出显示PID为 `2395910` (IPv4) 和 `2395917` (IPv6) 的 `docker-proxy` 进程占用了端口8000。

2.  **检查是否有活动的Docker容器映射该端口**：
    运行 `docker ps -a | grep ":8000->"` 来查看是否有容器明确将主机的8000端口映射到自身。
    在本次排查中，此命令未返回任何结果，表明没有活动的、已知的容器正在使用该端口映射。这可能意味着 `docker-proxy` 是一个残留进程。

3.  **停止占用端口的进程**：
    由于确认没有活动的容器直接映射此端口，我们决定停止这些 `docker-proxy` 进程：
    ```bash
    sudo kill 2395910 2395917
    ```
    **注意**：如果占用端口的是其他重要进程，您可能需要考虑停止该进程或将NIM服务映射到其他端口。

4.  **重新运行NIM启动脚本**：
    在停止冲突进程后，再次运行 `./scripts/03_run_nim_docker.sh`。

如果将来再次遇到此问题，可以参照以上步骤进行排查和解决。如果占用端口8000的进程是您无法或不想停止的重要服务，您可以编辑 `scripts/03_run_nim_docker.sh` 文件，修改 `-p 8000:8000` 为另一个可用的主机端口，例如 `-p 8001:8000`，然后通过新端口访问NIM服务。

### 4.4 与模型交互 (Python)

一旦NIM服务成功运行，我们可以使用Python脚本通过其OpenAI兼容的API接口来查询模型。

*   **创建 `src/query_nim_service.py` 文件:**
    该脚本使用 `openai` Python库与运行在 `http://localhost:8000/v1` 的NIM服务进行通信。
    脚本内容如下（或参考 `src/query_nim_service.py`）：
    ```python
    # src/query_nim_service.py
    from openai import OpenAI

    client = OpenAI(
      base_url = "http://localhost:8000/v1", # 如果修改了端口，这里也要改
      api_key = "none" # NIM 本地部署通常不需要 key
    )

    MODEL_TO_QUERY = "Llama-3.2-1B-custom-deployed" 
    # 上述模型名称应与您在 scripts/03_run_nim_docker.sh 中通过 NIM_SERVED_MODEL_NAME 设置的名称一致
    # 或者，如果NIM容器日志显示它以其他名称（如 meta-llama/Llama-3.2-1B）注册了模型，则使用该名称

    try:
        print(f"Attempting to query model: {MODEL_TO_QUERY} using /completions endpoint.")
        # 注意：基础模型通常没有聊天模板，因此我们使用 /completions 接口
        completion = client.completions.create(
          model=MODEL_TO_QUERY,
          prompt="What is the capital of France?", # 使用 prompt 参数
          temperature=0.7,
          top_p=1.0,
          max_tokens=50,
          stream=True
        )
        
        print(f"Response from {MODEL_TO_QUERY}:")
        for chunk in completion:
          if chunk.choices[0].text is not None: # 响应在 .text 中
            print(chunk.choices[0].text, end="")
        print("\n--- End of response ---")

    except Exception as e:
        print(f"Error querying model {MODEL_TO_QUERY}: {e}")
        # 如果遇到关于 chat template 的错误，说明模型不支持 /chat/completions 接口，
        # 或者您正在尝试使用该接口。请确保改用 /completions 接口，如上所示。
    ```

*   **安装依赖:**
    如果尚未安装 `openai` 库，请在您的Python环境中（推荐使用虚拟环境）安装：
    ```bash
    # 如果使用虚拟环境 (.venv)
    # source .venv/bin/activate 
    pip install openai
    ```

*   **执行脚本:**
    确保NIM Docker容器仍在运行。
    在**新的终端窗口**中，进入项目根目录并执行：
    ```bash
    # cd /path/to/your/MTG_ambassador/finetuning_model_deploy
    # source .venv/bin/activate # 如果使用了虚拟环境
    python src/query_nim_service.py
    ```
    记录输出和结果。

    **实际执行输出示例：**
    ```
    (.venv) cho@tut-server-11:~/workspace/MTG_ambassador/finetuning_model_deploy$ python src/query_nim_service.py
    Attempting to query model: Llama-3.2-1B-custom-deployed using /completions endpoint.
    Response from Llama-3.2-1B-custom-deployed:
     A. Rennes B. New York C. Des Moines D. Paris
    A. Rennes
    B. New York
    C. Des Moines
    D. Paris
    Answer: A
    Explanation:   Rennes is the capital of France.
    --- End of response ---
    ```
    *(注意：上述模型输出将 "Rennes" 错误地识别为法国的首都。这说明了基础模型可能存在的知识局限性或特定提示下的行为特性。实际应用中可能需要进一步的微调或提示工程来优化回答的准确性。)*

*   **预期问题与解决：`Chat template not found` 错误**
    如果您在尝试查询基础模型（如 `meta-llama/Llama-3.2-1B`）时，最初使用了 `client.chat.completions.create()` 方法（对应NIM的 `/v1/chat/completions` 接口），您可能会遇到类似以下的错误：
    ```
    Error querying model Llama-3.2-1B-custom-deployed: Error code: 500 - {'object': 'error', 'message': 'Model Llama-3.2-1B-custom-deployed does not have a default chat template defined in the tokenizer.  Set the `chat_template` field to issue a `/chat/completions` request or use `/completions` endpoint', ...}
    ```
    这是因为基础模型通常没有预定义的聊天模板，而 `/chat/completions` 接口需要它。
    **解决方案**：如上面的Python脚本示例所示，改用 `client.completions.create()` 方法（对应NIM的 `/v1/completions` 接口），并将 `messages` 参数替换为 `prompt` 参数。

## 5. 遇到的问题与解决方案

*   **问题1: 查询NIM服务时遇到 `Chat template not found` 错误**
    *   **发生情况:** 在使用 `src/query_nim_service.py` 脚本与通过NIM部署的 `meta-llama/Llama-3.2-1B` 模型交互时，如果最初尝试使用 `client.chat.completions.create()` 方法（对应NIM的 `/v1/chat/completions` 接口）。
    *   **相关日志/错误信息:**
        ```
        Error querying model Llama-3.2-1B-custom-deployed: Error code: 500 - {'object': 'error', 'message': 'Model Llama-3.2-1B-custom-deployed does not have a default chat template defined in the tokenizer.  Set the `chat_template` field to issue a `/chat/completions` request or use `/completions` endpoint', ...}
        ```
    *   **原因分析:** 基础版LLM（例如 `meta-llama/Llama-3.2-1B`）通常没有预定义的聊天模板（chat template）。NIM服务的 `/v1/chat/completions` API端点需要模型具备聊天模板才能正确处理对话格式的输入。
    *   **最终解决方案:** 修改Python查询脚本 (`src/query_nim_service.py`)，将API调用从 `/v1/chat/completions` 切换到 `/v1/completions`。
        *   具体操作：将 `client.chat.completions.create(...)` 调用更改为 `client.completions.create(...)`。
        *   参数调整：将 `messages=[{"role":"user", "content":"..."}]` 替换为 `prompt="用户的提问内容"`。
        *   此解决方案已在本文档 "4.4 与模型交互 (Python)" 部分的 `src/query_nim_service.py` 示例代码中详细说明和展示。
*   **问题2:** ...

## 6. 基于本地环境的调整与优化

在本节中，我们将详细记录在本地环境（NVIDIA RTX 6000 Ada Generation GPU，约48GB显存）下，针对 `meta-llama/Llama-3.2-1B` 模型部署所做的具体调整和优化尝试。一个关键的优化点是使用自定义的性能配置文件 (`NIM_MODEL_PROFILE`) 来指导NIM构建针对特定硬件和优化目标（如延迟或吞吐量）的TensorRT-LLM引擎。

### 6.1 使用自定义性能配置文件 (`NIM_MODEL_PROFILE`)

正如在 "4.3 NIM 微服务启动" 中关于性能配置文件的建议部分所述，NIM允许我们通过 `-e NIM_MODEL_PROFILE=<profile_name>` 参数手动指定一个性能配置文件。如果不指定，NIM会尝试自动选择。手动指定可以让我们更精确地控制优化方向。

**目标：** 探索并应用一个适合本地 RTX 6000 Ada 和 Llama-3.2-1B 模型的性能配置文件，以期获得更优的推理性能（例如，更低的延迟）。

**1. 查找适用的性能配置文件:**

   *   **查阅NVIDIA NIM官方文档:** 最权威的信息来源是NVIDIA官方文档。您需要查找与您的GPU架构（Ada Lovelace架构，类似于H100系列但有显存和核心数差异）和目标模型大小（1B参数级别）相关的可用配置文件。
   *   **理解配置文件命名约定:** 性能配置文件的名称通常包含以下信息：
        *   `tensorrt_llm`: 表明使用TensorRT-LLM后端。
        *   `h100` / `ada` / `a100` 等: 目标GPU架构或系列。您可能需要寻找与 `h100` 类似但适用于您具体显卡显存的配置，或者直接查找针对 `ada` 系列的配置（如果提供）。
        *   `bf16` / `fp16` / `int8` / `fp8`: 使用的精度（例如BF16、FP16、INT8量化、FP8量化）。
        *   `tp<N>`: 张量并行度 (Tensor Parallelism)。例如 `tp1`, `tp2`。
        *   `pp<N>`: 流水线并行度 (Pipeline Parallelism)。例如 `pp1`。
        *   `latency` / `throughput`: 优化目标。
   *   **NVIDIA博客示例:** 博客文章 ([Deploying Fine-tuned AI Models with NVIDIA NIM](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)) 中使用了 `tensorrt_llm-h100-bf16-tp2-pp1-latency` 作为 `OpenMath2-Llama3.1-8B` 在H100上的示例。对于我们的1B模型和RTX 6000 Ada，我们可能需要一个 `tp1` (因为模型较小，单GPU显存足够) 且针对延迟优化的配置文件。精度方面 `bf16` 或 `fp16` 是常见的选择。

   **您的调研和选择：**
   ```
   (请在此处记录您通过查阅文档或实验找到的，并决定尝试的 NIM_MODEL_PROFILE 值)
   例如: NIM_MODEL_PROFILE="tensorrt_llm_ada_bf16_tp1_pp1_latency" (这是一个假设的名称，请替换为实际调研结果)
   ```

   **`list-model-profiles` 命令执行结果 (针对 `nvcr.io/nim/meta/llama-3.2-1b-instruct:1`):**
   在 `2025-05-22` 尝试获取当前NIM镜像 (`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`) 支持的性能配置文件，通过在项目根目录下执行以下命令（确保已 `source MTG_ambassador/finetuning_model_deploy/scripts/01_set_env_vars.sh`）：
   ```bash
   geng idocker run --rm --gpus=all -e NGC_API_KEY=$NGC_API_KEY $IMG_NAME list-model-profiles
   ```
   得到的输出如下：
   ```text
   Environment variables set:
     NGC_API_KEY: nvapi-d5qS... (部分隐藏)
     NIM_CACHE_PATH: /tmp/nim/.cache
     MODEL_WEIGHT_PARENT_DIRECTORY: /home/cho/models
     IMG_NAME: nvcr.io/nim/meta/llama-3.2-1b-instruct:1
   
   REMINDER: Please ensure you have replaced 'YOUR_NGC_API_KEY_HERE' in this script with your actual NGC API Key.
   This script should be sourced: 'source ./scripts/01_set_env_vars.sh'
   
   ===========================================
   == NVIDIA Inference Microservice LLM NIM ==
   ===========================================
   
   NVIDIA Inference Microservice LLM NIM Version 1.8.4
   Model: meta/llama-3.2-1b-instruct
   
   Container image Copyright (c) 2016-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   
   This is a PLACEHOLDER LEGAL FILE. REPLACE WITH ACTUAL FILE.
   
   SYSTEM INFO
   - Free GPUs:
     -  [26b1:10de] (0) NVIDIA RTX 6000 Ada Generation[current utilization: 0%]
     -  [1fb1:10de] (1) NVIDIA T600[current utilization: 0%]
   2025-05-22T09:03:42.752024Z  INFO nim_hub_ngc::api::tokio::builder: ngc configured with api_loc: api.ngc.nvidia.com auth_loc: authn.nvidia.com scheme: https
   2025-05-22T09:03:43.669827Z  INFO nim_hub_ngc::api::tokio: Downloaded filename: config.json to blob: "/opt/nim/.cache/ngc/hub/models--nim--meta--llama-3.2-1b-instruct/blobs/c4c70d448a3481feffa70c72e7cc6501"
   2025-05-22T09:03:43.672925Z  INFO nim_hub_ngc::api::tokio::builder: ngc configured with api_loc: api.ngc.nvidia.com auth_loc: authn.nvidia.com scheme: https
   2025-05-22T09:03:43.673095Z  INFO nim_hub_ngc: File: config.json found in cache: "/opt/nim/.cache/ngc/hub/models--nim--meta--llama-3.2-1b-instruct/snapshots/hf-e9f8eff-nim1.5+/config.json"
   2025-05-22T09:03:43.673107Z  INFO nim_hub_ngc::api::tokio::public: Skipping download, using cached copy of file: config.json at path: "/opt/nim/.cache/ngc/hub/models--nim--meta--llama-3.2-1b-instruct/snapshots/hf-e9f8eff-nim1.5+/config.json"
   # ... (大量类似的 ngc 日志输出，此处省略以保持简洁) ...
   MODEL PROFILES
   - Compatible with system and runnable: <None>
   - Compilable to TRT-LLM using just-in-time compilation of HF models to TRTLLM engines: <None>
   - Incompatible with system:
     - 8b87146e39b0305ae1d73bc053564d1b4b4c565f81aa5abe3e84385544ca9b60 (tensorrt_llm-b200-fp8-tp1-pp1-throughput)
     - 7b508014e846234db3cabe5c9f38568b4ee96694b60600a0b71c621dc70cacf3 (tensorrt_llm-h100-fp8-tp1-pp1-throughput)
     - af876a179190d1832143f8b4f4a71f640f3df07b0503259cedee3e3a8363aa96 (tensorrt_llm-h200-fp8-tp1-pp1-throughput)
     - ad17776f4619854fccd50354f31132a558a1ca619930698fd184d6ccf5fe3c99 (tensorrt_llm-l40s-fp8-tp1-pp1-throughput)
     - 5811750e70b7e9f340f4d670c72fcbd5282e254aeb31f62fd4f937cfb9361007 (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput)
     - 222d1729a785201e8a021b226d74d227d01418c41b556283ee1bdbf0a818bd94 (tensorrt_llm-a100-bf16-tp1-pp1-throughput)
     - 74bfd8b2df5eafe452a9887637eef4820779fb4e1edb72a4a7a2a1a2d1e6480b (tensorrt_llm-a10g-bf16-tp1-pp1-throughput)
     - a4c63a91bccf635b570ddb6d14eeb6e7d0acb2389712892b08d21fad2ceaee38 (tensorrt_llm-b200-bf16-tp1-pp1-throughput)
     - e7dbd9a8ce6270d2ec649a0fecbcae9b5336566113525f20aee3809ba5e63856 (tensorrt_llm-h100-bf16-tp1-pp1-throughput)
     - 434e8d336fa23cbe151748d32b71e196d69f20d319ee8b59852a1ca31a48d311 (tensorrt_llm-h200-bf16-tp1-pp1-throughput)
     - ac5071bbd91efcc71dc486fcd5210779570868b3b8328b4abf7a408a58b5e57c (tensorrt_llm-l40s-bf16-tp1-pp1-throughput)
     - 25b5e251d366671a4011eaada9872ad1d02b48acc33aa0637853a3e3c3caa516 (tensorrt_llm-h100_nvl-bf16-tp1-pp1-throughput)
     - c6821c013c559912c37e61d7b954c5ca8fe07dda76d8bea0f4a52320e0a54427 (tensorrt_llm-a100_sxm4_40gb-bf16-tp1-pp1-throughput)
     - 4f904d571fe60ff24695b5ee2aa42da58cb460787a968f1e8a09f5a7e862728d (vllm-bf16-tp1-pp1)
     - bae7cf0b51c21f0e6f697593ee58dc8a555559b4b81903502a9e0ffbdc1b67a9 (tensorrt_llm-b200-fp8-tp1-pp1-throughput-lora-lora)
     - 018942e9a37a69b58ed514d7006ed851f57a1b2501ec4d39e3d862d8718893e5 (tensorrt_llm-h100-fp8-tp1-pp1-throughput-lora-lora)
     - 083e56197c3078f2932fc8d3d0e6d0d5b88d29ab8dbe154097deb322f227be07 (tensorrt_llm-h200-fp8-tp1-pp1-throughput-lora-lora)
     - 0782f55dcd12ec36d6126d9768fd364182986eecd25526eb206553df388057b7 (tensorrt_llm-l40s-fp8-tp1-pp1-throughput-lora-lora)
     - 22f2a8f33c93678bdfb831a62887f25642534525f1e0212f360b5490b03f99a2 (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-lora-lora)
     - 74eedaea9082525093f8346646709dc91efa7fd7ef2bc72d595feefaab94cf6f (tensorrt_llm-a100-bf16-tp1-pp1-throughput-lora-lora)
     - 0c85a1fe1aa38fa0f635a4c77bf25f7c94bfb9022564a929580415093f5b97d0 (tensorrt_llm-a10g-bf16-tp1-pp1-throughput-lora-lora)
     - a7baab3661c8376231396bf5529a78365892a33cc3416a11986fc2d96afcbcd8 (tensorrt_llm-b200-bf16-tp1-pp1-throughput-lora-lora)
     - 6e29cf433044b5a90f50d4cd1ca2e48d1ecb62eb5fa6fa9fb6f8317da61eebb3 (tensorrt_llm-h100-bf16-tp1-pp1-throughput-lora-lora)
     - f603fd56445cfe4f9448dfef2d9d1507465af46f1a0e60e9fb9796850bdbd7e6 (tensorrt_llm-h200-bf16-tp1-pp1-throughput-lora-lora)
     - 08662fae791cb76c40899ed539a7cbfdac61cad60d042220a3c321abd34a8a87 (tensorrt_llm-l40s-bf16-tp1-pp1-throughput-lora-lora)
     - 86b3fade693f8139fd213d70622d87dd6919f9f22dc7e8bbdfd84f1b5601d044 (tensorrt_llm-h100_nvl-bf16-tp1-pp1-throughput-lora-lora)
     - 47dbb6f7acda1fb4b1220d9462a66bda882e06e5d547e3b489c6bd7bf87df6e3 (tensorrt_llm-a100_sxm4_40gb-bf16-tp1-pp1-throughput-lora-lora)
     - f749ba07aade1d9e1c36ca1b4d0b67949122bd825e8aa6a52909115888a34b95 (vllm-bf16-tp1-pp1-lora)
     - ac34857f8dcbd174ad524974248f2faf271bd2a0355643b2cf1490d0fe7787c2 (tensorrt_llm-trtllm_buildable-bf16-tp1-pp1)
     - 7b8458eb682edb0d2a48b4019b098ba0bfbc4377aadeeaa11b346c63c7adf724 (tensorrt_llm-trtllm_buildable-bf16-tp1-pp1-lora)
   ```

   **分析与结论：**
   根据上述 `list-model-profiles` 命令的输出，针对我们使用的NIM镜像 `nvcr.io/nim/meta/llama-3.2-1b-instruct:1` 和本地GPU `NVIDIA RTX 6000 Ada Generation`：
   *   **没有直接兼容并可运行的预构建优化配置文件** (`Compatible with system and runnable: <None>`)。
   *   这意味着我们不能简单地从列表中选择一个现成的、保证最优的 `NIM_MODEL_PROFILE` 名称来填入启动脚本。
   *   NIM服务在启动时，如果没有指定有效的、兼容的 `NIM_MODEL_PROFILE`，它会尝试**自动检测硬件和模型，并动态构建一个优化的TensorRT-LLM引擎**。这个过程通常会利用类似 `tensorrt_llm-trtllm_buildable-bf16-tp1-pp1` 这样的通用构建策略（如列表末尾所示，尽管它们也被标记为"不兼容"，这可能是指不能直接作为 *预构建profile* 使用，但其构建逻辑可能被NIM内部采用）。
   *   因此，当前阶段的实验重点将是观察和记录NIM在不显式指定 `NIM_MODEL_PROFILE` （或使用其默认的构建行为）时的模型优化过程和性能表现。

**2. 修改NIM启动脚本:**

   基于以上分析，我们当前**不需要**在 `scripts/03_run_nim_docker.sh` 中显式设置或修改 `NIM_MODEL_PROFILE` 环境变量来指定一个特定的预构建配置文件（因为没有找到直接兼容的）。我们将依赖NIM的自动优化机制。

   如果您之前在 `scripts/03_run_nim_docker.sh` 中为实验添加了 `NIM_MODEL_PROFILE` 环境变量，请确保将其移除或注释掉，以恢复到NIM的默认行为。

   **您的脚本修改记录：**
   ```diff
   (如果您的 `scripts/03_run_nim_docker.sh` 中没有 `NIM_MODEL_PROFILE` 行，请在此注明"脚本未做修改，将使用NIM默认优化行为"。如果移除了相关行，请粘贴 diff。)
   ```

**3. 测试步骤:**
   现在，我们将聚焦于观察NIM在默认情况下的行为。

### 6.2 观察NIM默认优化行为与性能

在上一节中，我们发现针对 `meta-llama/Llama-3.2-1B` 模型和本地NVIDIA RTX 6000 Ada Generation GPU，NIM镜像 `nvcr.io/nim/meta/llama-3.2-1b-instruct:1` 并未提供直接兼容的预构建性能配置文件。因此，我们选择不显式设置 `NIM_MODEL_PROFILE` 环境变量，让NIM自行决定优化策略并动态构建TensorRT-LLM引擎。

**NIM服务启动与引擎构建日志分析 (执行 `./scripts/03_run_nim_docker.sh` 后)：**

以下关键信息从Docker容器日志中提取 (`docker logs <container_id>`)：

*   **启动时间与NIM版本：**
    *   服务启动尝试于 `2025-05-22 09:21:00` 左右。
    *   NVIDIA Inference Microservice LLM NIM Version `1.8.4`。
    *   模型识别为 `meta/llama-3.2-1b-instruct` (基于NIM镜像)。

*   **自动选择的性能配置文件：**
    *   NIM检测到2个兼容的配置文件，并最终选择：
        ```
        INFO 2025-05-22 09:21:08.183 ngc_injector.py:315] Selected profile: ac34857f8dcbd174ad524974248f2faf271bd2a0355643b2cf1490d0fe7787c2 (tensorrt_llm-trtllm_buildable-bf16-tp1-pp1)
        ```
    *   **所选配置文件元数据：**
        *   `llm_engine: tensorrt_llm`
        *   `pp: 1` (Pipeline Parallelism = 1)
        *   `precision: bf16` (BF16精度)
        *   `tp: 1` (Tensor Parallelism = 1)
        *   `trtllm_buildable: true` (表明这是一个用于动态构建TensorRT-LLM引擎的配置)

*   **TensorRT-LLM 引擎构建过程：**
    *   NIM开始为模型 `/opt/weights/hf/Llama-3.2-1B` (这是我们在 `scripts/03_run_nim_docker.sh` 中通过 `NIM_FT_MODEL` 指定的本地权重路径，在容器内映射到此路径) 构建TensorRT-LLM引擎。
    *   构建参数包括：`max_seq_len=131072`, `dtype=torch.bfloat16`, `tensor_parallel_size=1`, `pipeline_parallel_size=1` 等。
    *   **引擎构建耗时：**
        *   检查点读取与转换耗时：`INFO 2025-05-22 09:21:35.961 build_utils.py:239] Total time of reading and converting rank0 checkpoint: 0.18 s`
        *   引擎构建耗时 (rank0)：`INFO 2025-05-22 09:22:59.751 build_utils.py:247] Total time of building rank0 engine: 83.79 s`
        *   因此，TensorRT-LLM引擎构建总共花费了大约 **83.8 秒**。

*   **引擎加载与服务启动：**
    *   引擎成功加载：`Engine size in bytes 3112711548` (约 2.89 GB)。
    *   NIM服务成功启动，并在 `http://0.0.0.0:8000` 上监听。
        ```
        INFO 2025-05-22 09:23:01.032 launcher.py:298] Worker starting (pid: 212)
        INFO 2025-05-22 09:23:02.182 server.py:89] Starting HTTP server on port 8000
        INFO 2025-05-22 09:23:02.182 server.py:91] Listening on address 0.0.0.0
        INFO 2025-05-22 09:23:02.202 server.py:162] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

*   **模型查询测试：**
    *   如 "4.4 与模型交互 (Python)" 部分所述，服务启动后，使用 `src/query_nim_service.py` 脚本成功通过 `/v1/completions` 端点查询了模型。相关日志条目：
        `INFO 2025-05-22 09:28:45.91 httptools_impl.py:481] ::1:33148 - "POST /v1/completions HTTP/1.1" 200`

**总结：**
实验结果表明，即使在没有为特定硬件 (NVIDIA RTX 6000 Ada) 和NIM镜像 (`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`) 找到预构建的、直接兼容的优化配置文件 (`NIM_MODEL_PROFILE`) 的情况下，NVIDIA NIM也能：
1.  自动选择一个合适的构建策略（在此案例中为 `tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`）。
2.  利用此策略，动态地为提供的模型权重 (`meta-llama/Llama-3.2-1B`) 构建一个针对当前GPU优化的TensorRT-LLM引擎。
3.  整个引擎构建过程在本地环境中耗时约1分半钟。
4.  成功启动服务并能正确响应API请求。

这充分展示了NIM在简化部署流程和自动化优化方面的能力，使其在多样化的硬件和模型组合下依然具有良好的适用性。

## 7. 总结与发表要点

*   **核心收获:**
    *   **成功部署与验证：** 成功在本地NVIDIA RTX 6000 Ada (约48GB显存) 环境下，使用NVIDIA NIM部署了 `meta-llama/Llama-3.2-1B` 模型，并验证了其服务可用性。
    *   **理解NIM工作流：** 对NIM的工作流程有了更深入的理解，特别是其能够自动检测硬件、选择合适的构建策略（如 `tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`），并动态为本地模型权重构建优化的TensorRT-LLM引擎（耗时约1.5分钟）。
    *   **部署简化性：** 体会到NIM结合TensorRT-LLM在简化LLM部署方面的强大能力，大大降低了手动配置和优化推理引擎的复杂性。
    *   **标准化接口价值：** NIM提供的OpenAI兼容API接口使得与已部署模型的交互和集成变得非常便捷。

*   **关键挑战与应对:**
    *   **GPU显存限制：** 本地GPU（约48GB）远小于博客示例环境（80GB），最初担心无法运行。通过选择参数量较小的 `meta-llama/Llama-3.2-1B` 模型，成功在此环境下完成了部署和测试。
    *   **环境变量与路径配置：**
        *   多次遇到因环境变量（如 `IMG_NAME`, `MODEL_WEIGHT_PARENT_DIRECTORY`）未正确加载或脚本中路径（如 `01_set_env_vars.sh` 的相对路径）解析不当导致命令执行失败。通过确保 `source` 脚本、使用明确的绝对路径（如修改 `01_set_env_vars.sh` 中 `PROJECT_ROOT_DIR` 的确定方式）以及注意命令执行的当前工作目录来解决。
    *   **端口冲突：** 首次启动NIM Docker容器时，遇到了主机端口8000已被占用的问题。通过 `sudo netstat -tulnp | grep :8000` 或 `sudo ss -tulnp | grep :8000` 查找到占用进程（通常是残留的 `docker-proxy`），并使用 `sudo kill <PID>` 将其停止后解决。
    *   **模型API使用差异 (`Chat template not found`)：** 在使用Python脚本查询NIM服务部署的 `meta-llama/Llama-3.2-1B` 基础模型时，最初尝试使用 `/v1/chat/completions` 接口（对应 `client.chat.completions.create()`）导致了 "Chat template not found" 错误。原因是基础模型通常没有预设聊天模板。通过切换到 `/v1/completions` 接口（对应 `client.completions.create()`）并使用 `prompt` 参数代替 `messages` 参数成功解决。
    *   **Docker与GPU驱动兼容性：** 虽然本次未直接遇到，但需要注意Docker、NVIDIA驱动、CUDA版本之间的兼容性，这是NIM服务正常运行的基础。

*   **未来可探索方向:**
    *   **不同模型与微调测试：** 尝试部署其他不同参数量或类型的模型，特别是经过SFT（Supervised Fine-tuning）的微调模型，以更全面地验证NIM对不同微调模型的支持程度和优化效果。
    *   **性能优化策略探索：** 如果硬件支持且NIM提供了兼容的性能配置文件，可以测试更激进的优化策略，例如FP8量化（如果RTX 6000 Ada支持且NIM image包含相应profile），并比较其对性能和模型准确率的影响。
    *   **细致化性能基准测试：** 进行更系统、细致的性能基准测试，使用标准化工具（如 `triton-perf-analyzer` 或自定义脚本）记录和对比在不同配置（例如，有无显式 `NIM_MODEL_PROFILE`、不同模型、不同并发量）下的推理延迟、吞吐量（TPS, Tokens Per Second）和GPU资源利用率。
    *   **多GPU与分布式部署：** 研究NIM在多GPU环境下的部署配置（如张量并行TP、流水线并行PP的设置）和性能扩展性。
    *   **应用集成与工作流：** 探索如何将通过NIM部署的模型服务更紧密地集成到复杂的人工智能应用、RAG系统或Magentic Agent工作流中。

*   **给其他尝试者的建议:**
    *   **仔细阅读官方文档：** 在开始之前，务必仔细研读NVIDIA NIM的官方文档，特别是关于支持的模型列表、NIM容器镜像版本的选择（通常与基础模型对应）、所需环境变量的详细说明以及硬件兼容性要求。
    *   **重视环境变量配置：** 确保所有必要的环境变量（如 `NGC_API_KEY`, `NIM_CACHE_PATH`, `MODEL_WEIGHT_PARENT_DIRECTORY`, `IMG_NAME` 等）在执行相关脚本的shell会话中都已正确设置和导出。强烈推荐使用项目提供的 `scripts/01_set_env_vars.sh` 脚本进行统一管理，并切记在每次新的终端会话或执行Docker相关命令前 `source` 该脚本。
    *   **善用Docker日志排错：** 当遇到Docker容器启动失败、服务行为异常或模型加载错误时，应首先使用 `docker ps -a` 查看容器的运行状态，然后立即使用 `docker logs <container_id_or_name>` (可以加上 `-f` 参数实时查看) 仔细检查容器内部的日志输出。这通常能提供最直接、最详细的错误信息和故障线索。
    *   **从小处着手，逐步验证：** 特别是在显存等硬件资源相对受限的本地环境中，建议从参数量较小的模型（如本项目中的1B模型）开始尝试，确保基础流程跑通后，再逐步挑战更大的模型或更复杂的配置。
    *   **理解API端点差异：** 清晰区分基础大模型与经过指令/对话微调的模型在通过API查询时的行为和接口差异。基础模型通常使用 `/v1/completions` 端点和 `prompt` 参数，而指令/对话模型更适合 `/v1/chat/completions` 端点和 `messages` 参数。错误的使用可能导致 "Chat template not found" 等问题。
    *   **确保模型权重路径正确：** `NIM_FT_MODEL` 环境变量指定的是Docker容器内部的模型权重路径（通常是 `/opt/weights/hf/<model_folder_name>`），要确保通过 `-v $MODEL_WEIGHT_PARENT_DIRECTORY:/opt/weights/hf` 正确地将包含模型文件夹的本地父目录挂载到了容器的 `/opt/weights/hf`。

---
*文档将持续更新。* 
# ACE Agent llm_bot 示例部署步骤总结

本文档总结了部署 NVIDIA ACE Agent `llm_bot` 示例所执行的操作步骤和文件更改。

## 1. 系统环境检查

*   **目的:** 评估系统是否满足部署 NVIDIA ACE/Tokkio 的基本要求。
*   **命令:**
    ```bash
    (lsb_release -a || cat /etc/os-release) && echo '---' && (nvidia-smi || echo 'nvidia-smi not found') && echo '---' && free -h && echo '---' && df -h / && echo '---' && (docker --version || echo 'docker not found') && echo '---' && (kubectl version --client --short || echo 'kubectl not found')
    ```
*   **结果:**
    *   操作系统: Ubuntu 20.04.6 LTS (注意: Tokkio 要求 22.04)
    *   GPU: NVIDIA RTX 6000 Ada Generation (48GB), NVIDIA T600 (4GB) (满足要求)
    *   内存: 251Gi (充足)
    *   磁盘: 1.8T 总量, 205G 可用 (注意: Tokkio 建议 >700GB)
    *   Docker: 已安装 (版本 28.0.4)
    *   Kubernetes (kubectl): 未安装 (Tokkio 需要, ACE Agent Docker Compose 示例不需要)
*   **结论:** 系统适合部署 ACE Agent Docker Compose 示例，但部署完整 Tokkio 工作流程存在障碍 (OS 版本, 磁盘空间, K8s)。

## 2. 准备 ACE Agent 代码

*   **目的:** 获取 ACE Agent 的代码和示例。
*   **命令:**
    ```bash
    # 克隆仓库 (如果不存在) 并进入目标目录
    if [ ! -d "ACE" ]; then git clone https://github.com/NVIDIA/ACE.git; fi && cd ACE/microservices/ace_agent/4.1 && pwd
    ```
*   **当前目录:** `/home/cho/workspace/ACE/microservices/ace_agent/4.1`

## 3. 配置 `llm_bot` 使用托管 NIM

*   **目的:** 修改示例代码，使其连接到 NVIDIA API Catalog 提供的托管 LLM 模型，而不是尝试在本地运行 LLM。
*   **文件:** `ACE/microservices/ace_agent/4.1/samples/llm_bot/actions.py`
*   **更改:**
    *   注释掉了本地 NIM (NeMo Inference Microservice) 的 `AsyncOpenAI` 客户端配置。
    *   启用了使用 `https://integrate.api.nvidia.com/v1` 作为 `base_url` 并通过 `NVIDIA_API_KEY` 环境变量进行认证的 `AsyncOpenAI` 客户端配置。

## 4. 部署 Riva ASR/TTS 模型

*   **目的:** 下载并准备语音识别 (ASR) 和语音合成 (TTS) 模型，这是语音交互的基础。
*   **方法:** 使用 Docker Compose 和 `model-utils-speech` 服务。
*   **尝试 1 & 2 (失败):**
    *   命令: `export BOT_PATH=./samples/llm_bot/ && source deploy/docker/docker_init.sh` 后接 `docker compose -f deploy/docker/docker-compose.yml up model-utils-speech`
    *   问题: 第一次因路径错误失败。第二次因 NGC 认证失败 (`Invalid org`, `NGC_CLI_API_KEY` 环境变量未传递给容器)。
*   **尝试 3 (成功):**
    *   **操作:** 创建了文件 `ACE/microservices/ace_agent/4.1/deploy/docker/.env` 并指导用户将 `NGC_CLI_API_KEY` 和 `NVIDIA_API_KEY` 填入其中。
        ```
        # Required for downloading ACE container images and models from NGC
        NGC_CLI_API_KEY="YOUR_NGC_API_KEY_HERE" # 用户已替换

        # Required when using NVIDIA API Catalog hosted NIM models (as configured in actions.py)
        NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY_FROM_API_CATALOG_HERE" # 用户已替换
        ```
    *   **命令:**
        ```bash
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && docker compose -f deploy/docker/docker-compose.yml up model-utils-speech
        ```
    *   **结果:** 成功通过认证，下载了 ASR (parakeet-1.1b) 和 TTS (fastpitch_hifigan) 模型。使用 ServiceMaker 对模型进行了处理，包括构建 TensorRT 引擎。最后，启动了 Riva Speech Server 并加载了模型。`model-utils-speech` 容器成功完成并退出。

## 5. 部署 ACE Agent 核心服务 (`speech-event-bot`)

*   **目的:** 启动包含 Chat Engine, Chat Controller, Web UI 等核心组件的服务。
*   **尝试 1 (失败):**
    *   命令: `cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && docker compose -f deploy/docker/docker-compose.yml up --build speech-event-bot -d`
    *   问题: `docker_init.sh` 未在同一会话执行，导致环境变量 (如 `TAG`, `DOCKER_REGISTRY`) 丢失，出现 `invalid reference format` 错误，无法找到正确的镜像名称。
*   **尝试 2 (成功):**
    *   **命令:**
        ```bash
        # 确保在运行 up 命令前执行 docker_init.sh
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && source deploy/docker/docker_init.sh && echo 'Environment re-initialized.' && docker compose -f deploy/docker/docker-compose.yml up --build speech-event-bot -d
        ```
    *   **结果:** 成功构建了相关镜像并启动了所有必需的容器 (`chat-engine-event-speech`, `chat-controller`, `nlp-server`, `plugin-server`, `redis-server`, `bot-web-ui-client`, `bot-web-ui-speech-event`)。

## 6. 测试与故障排查

*   **目的:** 验证部署是否成功并解决遇到的问题。
*   **问题 1 (Web UI 错误 & 无法回答):**
    *   **症状:** 访问 Web UI (`http://10.204.222.147:7006/`) 后，出现错误 `A fatal error occurred while running task GRPCSpeechTask. Message: "[canceled] This operation was aborted".` 并且机器人无法回答问题。
    *   **日志诊断:**
        *   `chat-controller` 日志显示无法连接 Riva ASR 服务: `Unable to establish connection to server localhost:50051`。
        *   `chat-engine-event-speech` 日志显示无法加载 Bot: `CRITICAL No bots available. Please register a bot by providing the path to its config folder.`
    *   **原因分析:**
        *   `chat-engine` 错误是由于 `BOT_PATH` 环境变量未能正确传递给容器。
        *   `chat-controller` 错误原因尚不完全明确，但表明语音服务链路上存在问题。
    *   **解决方案尝试:** 停止服务后，在启动命令前显式设置 `BOT_PATH` 并重新运行 `docker_init.sh`，然后再次启动 `speech-event-bot`。
        ```bash
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && \
        export BOT_PATH=./samples/llm_bot/ && \
        source deploy/docker/docker_init.sh && \
        echo "BOT_PATH is set to: $BOT_PATH" && \
        docker compose -f deploy/docker/docker-compose.yml up --build speech-event-bot -d
        ```
    *   **结果:** 服务成功启动，**文本输入功能恢复**。
*   **问题 2 (无法启用麦克风):**
    *   **症状:** 在 Web UI 中，无法点击或启用麦克风进行语音输入。
    *   **原因分析:** 浏览器安全策略阻止通过不安全的 HTTP 连接 (`http://10.204.222.147:7006`) 访问麦克风。
    *   **解决方案:** 修改浏览器设置，将该地址视为安全来源。
        1.  访问 `chrome://flags` 或 `edge://flags`。
        2.  搜索并启用 `#unsafely-treat-insecure-origin-as-secure` 标志。
        3.  在文本框中添加 `http://10.204.222.147:7006`。
        4.  重启浏览器。
    *   **结果:** **麦克风成功启用**。

## 7. 当前状态和下一步

*   **当前状态:** 所有服务已成功启动。Web UI 的**文本和语音输入功能均已启用**并可进行测试。
*   **下一步:**
    1.  全面测试 Web UI 的文本和语音交互功能。
    2.  (可选) 探索其他 ACE Agent 功能或示例。
    3.  完成测试后，停止所有服务: `cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && docker compose -f deploy/docker/docker-compose.yml down`。
*   **重启服务:** 如果需要重启所有 ACE Agent 服务，请执行以下步骤：
    1.  **停止当前服务 (如果正在运行):**
        ```bash
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && docker compose -f deploy/docker/docker-compose.yml down
        ```
    2.  **重新启动服务:**
        ```bash
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && \
        export BOT_PATH=./samples/llm_bot/ && \
        source deploy/docker/docker_init.sh && \
        docker compose -f deploy/docker/docker-compose.yml up --build speech-event-bot -d
        ```

## 8. 模型部署和 API 使用情况

### LLM 模型
*   **部署方式:** NVIDIA API Catalog 托管的 NIM 服务
*   **使用模型:** `meta/llama3-8b-instruct`
*   **配置信息:**
    ```python
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    MODEL = "meta/llama3-8b-instruct"
    TEMPERATURE = 0.2
    TOP_P = 0.7
    MAX_TOKENS = 1024
    SYSTEM_PROMPT = "You are a helpful AI assistant."
    ```
*   **认证方式:** 使用 `NVIDIA_API_KEY` 环境变量
*   **特点:**
    - 支持流式输出，降低响应延迟
    - 支持对话历史管理
    - 平均每个用户查询可能需要 2-3 次 API 调用（由于 240ms 暂停重触发机制）

### 语音模型（本地部署）=
*   **ASR (语音识别):**
    - 模型: Riva ASR `parakeet-ctc-1.1b` 英文模型
    - 特点:
        * 支持低延迟的 ASR 2 pass End of Utterance (EOU)
        * 支持单词权重提升（word boosting）功能
        * 支持脏话过滤
        * 对英语口音具有较好的鲁棒性
*   **TTS (语音合成):**
    - 模型: Riva TTS `fastpitch_hifigan` 英文模型
    - 通过 gRPC APIs 提供服务
    - 支持实时语音合成

### 部署架构特点
1. **混合部署策略:**
   - LLM：使用云端 API（降低计算资源需求）
   - 语音服务：本地部署（确保低延迟）

2. **优化措施:**
   - 使用 NVIDIA TensorRT 进行模型优化
   - 通过 Triton Inference Server 进行模型部署
   - Chat Controller 优化确保低延迟和高吞吐量

3. **可选配置:**
   - 支持切换到完全本地部署（需要 A100 或 H100 GPU）
   - 支持使用 OpenAI API 替代 NVIDIA NIM
   - 支持使用第三方 TTS 解决方案（如 ElevenLabs）

4. **系统集成:**
   - 使用 gRPC APIs 实现流式低延迟语音服务
   - 支持 REST APIs 用于纯文本聊天机器人
   - 支持与 LangChain、LlamaIndex 等框架的集成 

## 9. 支持日语的修改分析

本章节分析了在当前 Docker Compose 部署的 `llm_bot` 基础上，支持日语交互所需的修改步骤和注意事项。

### 核心策略

ACE Agent 支持多语言部署，但一个实例通常配置为处理一种主要语言。支持日语主要有两种策略：

1.  **使用多语言或日语 LLM:** 直接使用能理解和生成日语的 LLM 模型。
2.  **使用神经机器翻译 (NMT):** 将日语输入翻译成 LLM 能理解的语言（如英语），然后将 LLM 的回复翻译回日语。

### 具体修改步骤 (基于 Docker Compose)

1.  **检查并部署日语 Riva 模型:**
    *   **ASR (语音识别):**
        *   检查 Riva 文档或 NGC 目录确认是否有可用的日语 ASR 模型 (例如 `ja-JP` 相关模型)。
        *   如果找到，修改 `model-utils-speech` 服务的配置（如 `docker_init.sh` 或 `docker-compose.yml`），将英文 ASR 模型替换为日语模型。
    *   **TTS (语音合成):**
        *   检查 Riva 是否提供日语 TTS 模型和语音名称。
        *   如果找到，修改 `model-utils-speech` 配置，替换英文 TTS 模型为日语模型。
        *   **替代方案:** 如果 Riva 不支持日语 TTS，可考虑集成第三方 TTS 服务 (如 ElevenLabs)，需修改 `nlp-server`。
    *   **NMT (神经机器翻译 - 如果采用策略 2):**
        *   检查 Riva 是否提供支持日语翻译的 NMT 模型。
        *   如果选择 NMT 策略，需将 NMT 模型添加到 `model-utils-speech` 的部署列表中。

2.  **修改 Chat Controller 配置:**
    *   **文件:** `ACE/microservices/ace_agent/4.1/deploy/docker/docker-compose.yml` 中 `chat-controller` 服务的环境变量或其引用的配置文件。
    *   **ASR 语言:** 修改相关配置项 (如 `RIVA_ASR_LANGUAGE_CODE`) 从 `"en-US"` 改为 `"ja-JP"`。
    *   **TTS 语言和语音:** 修改相关配置项 (如 `RIVA_TTS_LANGUAGE_CODE` 和 `RIVA_TTS_VOICE_NAME`) 为日语代码和可用的语音名称。

3.  **修改 LLM 配置或 Bot 逻辑 (`actions.py`):**
    *   **策略 1 (使用日语/多语言 LLM):**
        *   **文件:** `ACE/microservices/ace_agent/4.1/samples/llm_bot/actions.py`
        *   修改 `MODEL` 变量为支持日语的 LLM 标识符。
        *   如果需要，更新 `BASE_URL` 和认证方式。
        *   调整 `SYSTEM_PROMPT`。
    *   **策略 2 (使用 NMT):**
        *   **文件:** `ACE/microservices/ace_agent/4.1/samples/llm_bot/actions.py`
        *   保持 LLM 配置不变。
        *   修改 `call_nim_local_llm` action：
            *   在调用 LLM 前，调用 Riva NMT 将日语查询翻译成英文。
            *   使用翻译后的英文调用 LLM。
            *   在返回结果前，调用 Riva NMT 将 LLM 的英文回复翻译回日语。
            *   这需要了解如何通过 gRPC 调用 Riva NMT 服务。

4.  **重建并重启服务：**
    *   修改配置和代码后，停止现有服务:
        ```bash
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && docker compose -f deploy/docker/docker-compose.yml down
        ```
    *   根据新的模型配置更新 `docker_init.sh` (如果需要)，然后重新启动服务:
        ```bash
        cd /home/cho/workspace/ACE/microservices/ace_agent/4.1 && export BOT_PATH=./samples/llm_bot/ && source deploy/docker/docker_init.sh && docker compose -f deploy/docker/docker-compose.yml up --build speech-event-bot -d
        ```

### 重要注意事项
*   **模型可用性:** 确认 Riva 是否提供高质量的日语 ASR 和 TTS 模型是关键。
*   **LLM 能力:** 当前使用的 Llama3 8B 可能不足以很好地处理日语，需要选择合适的多语言或日语 LLM。
*   **单语言实例:** 修改后，该部署实例将主要处理日语。 
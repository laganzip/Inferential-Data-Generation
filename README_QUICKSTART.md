# Inferential Data Generation 快速入门

## 1. 工程功能

`inferential_data_generation` 用来生成“可解释的时序推理数据”。

核心流程是：
1. 场景生成器先合成一个完整时序样本（历史真实值 + 未来真实值 + 协变量 + 事件信息）。
2. 初始预测器先给出一版未来预测（`initial_prediction_values`）。
3. 程序计算真实修正量（`correction_delta_values = true - initial_prediction`）。
4. 可选：再调用外部 LLM 生成“模型修正量”（`llm_correction_delta_values`）。
5. 最终输出 JSONL，并可画图对比“真实值 / 初始预测 / 修正后预测”。

## 2. 代码结构（先看这几个入口就够）

1. `inferential_data_generation/cli.py`
用途：基础数据集生成入口（不做 LLM 修正）。

2. `inferential_data_generation/generate_llm_correction_demo.py`
用途：生成数据并追加 LLM 修正结果，或读取现有 JSONL 只做修正。

3. `inferential_data_generation/visualize_demo.py`
用途：画“真实值 vs 初始预测 vs 真实修正后曲线”。

4. `inferential_data_generation/visualize_llm_correction_demo.py`
用途：画“真实值 vs 初始预测 vs LLM 修正后曲线”。

5. 🌟`inferential_data_generation/scripts/generate_temporal_reasoning_example.sh`
用途：基础示例脚本（数据生成 + 可视化）。

6. 🌟`inferential_data_generation/scripts/test_llm_correction_deepseek3_2_example.sh`
用途：修正示例脚本（生成/评测 + LLM 修正 + 可视化）。

## 3. 支持的核心能力

### 3.1 场景（scene）

当前内置 3 个场景：
1. `temporal_physical_event`：建筑冷站负荷 vs 室外气温
2. `data_center_cooling_event`：数据中心冷却功率 vs 机房进风温度
3. `commercial_hvac_event`：商业综合体空调负荷 vs 体感温度指数

实现位置：`inferential_data_generation/scenes/temporal_physical_event.py`

### 3.2 数据类型（data-type）

1. `ts_cov_text`：包含协变量序列、上下文文本、结构化事件信息。
2. `ts_target_only`：只保留目标序列相关字段。

实现位置：`inferential_data_generation/data_types/`

### 3.3 初始预测器（predictor）

1. `chronos2`：优先用 Chronos2；不可用时可按配置回退。
2. `heuristic`：启发式预测，无外部大模型依赖。

实现位置：`inferential_data_generation/predictors/`

### 3.4 两类 LLM 能力

1. 上下文文本生成（`context_generation_mode=llm`）
作用：把结构化事件写成自然中文上下文。

2. 修正量生成（LLM correction）
作用：让 LLM 输出分段修正，再展开成完整 `llm_correction_delta_values`。

## 4. 运行前准备（最小必需）

### 4.1 基本环境

1. 在仓库根目录执行命令。
2. 可用 `python3`。
3. 安装依赖包（见下方“必要依赖清单”）。
4. 如果你要用 `chronos2`，还要保证 Chronos2 运行时和模型目录可用。

### 4.1.1 必要依赖清单

按代码实际 import，基础流程（生成 + 可视化 + LLM 请求）的必要第三方依赖是：

1. `numpy`：数值计算与绘图数据处理。
2. `matplotlib`：生成可视化图。
3. `requests`：调用上下文 LLM 和修正 LLM 的 HTTP API。

如果你使用 `PREDICTOR=chronos2`，还需要：

1. `torch`：Chronos2 推理依赖。
2. Chronos2 运行代码（需要额外加载安装）。

### 4.1.2 tpt 环境实测版本

建议版本如下：

1. `numpy==2.1.2`
2. `matplotlib==3.10.7`
3. `requests==2.32.5`
4. `torch==2.6.0+cu118`

你可以用同一条命令复查：

```bash
conda run -n tpt python -c "import importlib.metadata as md; pkgs=['numpy','matplotlib','requests','torch']; [print(f'{p}=={md.version(p)}') for p in pkgs]"
```

### 4.2 推荐先配 `.env`

脚本会自动读取仓库根目录 `.env`。你可以放这些变量（按需）：

```bash
# 上下文生成 LLM（可选）
CONTEXT_LLM_BASE_URL=
CONTEXT_LLM_API_KEY=
CONTEXT_LLM_MODEL=

# 修正 LLM（可选）
CORRECTION_LLM_BASE_URL=
CORRECTION_LLM_API_KEY=
CORRECTION_LLM_MODEL=

# Chronos2（按需）
CHRONOS2_MODEL_PATH=
```

注意：示例脚本默认对关键项做“缺值即报错”，不会静默兜底。

## 5. 示例脚本 1：基础生成与可视化

脚本：`inferential_data_generation/scripts/generate_temporal_reasoning_example.sh`

### 5.1 这个脚本做什么

1. 调 `python -m inferential_data_generation.cli` 生成 JSONL。
2. 调 `inferential_data_generation/visualize_demo.py` 画图。

### 5.2 最小可运行配置

```bash
export PLOT_PYTHON=/path/to/your/python
export CHRONOS2_MODEL_PATH=/path/to/Chronos2
export LLM_API_KEY=your_api_key
export LLM_MODEL=deepseek-v3.2
# 可选
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

bash inferential_data_generation/scripts/generate_temporal_reasoning_example.sh
```

如果你不想用上下文 LLM，改成：

```bash
export CONTEXT_GENERATION_MODE=template
bash inferential_data_generation/scripts/generate_temporal_reasoning_example.sh
```

### 5.3 关键参数

1. `SCENE`：场景名，默认 `temporal_physical_event`。
2. `DATA_TYPE`：默认 `ts_cov_text`。
3. `NUM_SAMPLES`：生成样本数。
4. `PREDICTOR`：`chronos2` 或 `heuristic`。
5. `ALLOW_PREDICTOR_FALLBACK`：`0` 失败即停，`1` 允许回退。
6. `ALLOW_CONTEXT_FALLBACK`：`0` 失败即停，`1` 允许回退模板文本。

### 5.4 输出位置

1. 数据：`DATA_OUTPUT`（默认 `data/temporal_physical_event_demo_*.jsonl`）。
2. 图片：`PLOT_OUTPUT_DIR`（默认 `data/generate_temporal_reasoning_plots`）。

## 6. 示例脚本 2：LLM 修正流程

脚本：`inferential_data_generation/scripts/test_llm_correction_deepseek3_2_example.sh`

### 6.1 这个脚本做什么

1. 先生成样本或读取已有样本。
2. 调 `generate_llm_correction_demo.py` 生成 `llm_correction_delta_values`。
3. 调 `visualize_llm_correction_demo.py` 画修正效果图。

### 6.2 三种模式

1. `MODE=generate`：只生成新样本并修正。
2. `MODE=evaluate`：读取 `INPUT_DATA` 后做修正。
3. `MODE=both`：默认含义是走完整流程（先生成再修正）。

### 6.3 最小可运行配置（走 evaluate）

```bash
export PLOT_PYTHON=/path/to/your/python
export INPUT_DATA=/abs/path/to/existing.jsonl
export CHRONOS2_MODEL_PATH=/path/to/Chronos2

export CONTEXT_LLM_API_KEY=your_context_key
export CONTEXT_LLM_MODEL=glm-4-flash-250414
# 可选
export CONTEXT_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

export CORRECTION_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export CORRECTION_LLM_API_KEY=your_correction_key
export CORRECTION_LLM_MODEL=deepseek-v3.2

bash inferential_data_generation/scripts/test_llm_correction_deepseek3_2_example.sh
```

### 6.4 只验证前半流程（跳过修正）

```bash
export SKIP_LLM_CORRECTION=1
bash inferential_data_generation/scripts/test_llm_correction_deepseek3_2_example.sh
```

说明：`SKIP_LLM_CORRECTION=1` 时不会生成 `llm_corrected_prediction_values`，脚本也会跳过可视化。

### 6.5 关键参数

1. `CORRECTION_LLM_*`：修正模型配置。
2. `ALLOW_CORRECTION_FALLBACK`：修正失败是否允许降为零修正。
3. `CORRECTION_LLM_MAX_RETRIES`：修正请求失败后的重试次数。
4. `NUM_SAMPLES`：生成样本数。
5. `SEED`：随机种子，保证可复现。

## 7. 结果文件里你会看到什么

### 7.1 基础生成 JSONL 常见字段

1. `target_name`：目标变量名。
2. `context_name`：协变量名。
3. `history_target_values` / `future_target_values`：真实历史/未来。
4. `initial_prediction_values`：初始预测。
5. `correction_delta_values`：真实修正量（true - initial）。
6. `context_description`：上下文文本（若启用）。
7. `structured_event_context`：结构化事件列表（若启用）。

### 7.2 LLM 修正 JSONL 额外字段

1. `llm_correction_delta_values`：LLM 生成的修正量。
2. `llm_corrected_prediction_values`：`initial + llm_correction_delta`。
3. `llm_correction_reasoning_chain`：LLM 修正解释。
4. `llm_correction_generation_status`：`ok` 或 `fallback`。
5. `llm_correction_error`：失败时的错误信息（可能存在）。

## 8. 新手最常见报错与处理

1. 报错：`PLOT_PYTHON 不可执行`
处理：把 `PLOT_PYTHON` 改成可执行解释器绝对路径。

2. 报错：`CHRONOS2_MODEL_PATH 目录不存在`
处理：填真实模型目录，或改 `PREDICTOR=heuristic` 先跑通。

3. 报错：`MODE=evaluate 时必须设置 INPUT_DATA`
处理：给 `INPUT_DATA` 传已存在 JSONL 绝对路径。

4. 报错：缺 `*_API_KEY` 或 `*_MODEL`
处理：补齐对应 LLM 配置，或把相关流程切到 template/skip 模式。

5. 报错：`Record does not contain 'llm_corrected_prediction_values'`
处理：你还没跑修正步骤，或设置了 `SKIP_LLM_CORRECTION=1`。

## 9. 二次开发的最短路径

1. 想新增业务场景：在 `scenes/` 新增类并注册到 `scenes/__init__.py`。
2. 想新增输出格式：在 `data_types/` 新增 assembler 并注册。
3. 想新增预测器：在 `predictors/` 新增 predictor 并注册。
4. 想改 LLM 修正规则：改 `llm_correction.py` 的 prompt 和解析逻辑。

## 10. 一条建议

第一次接手时，不要一上来就开 LLM 全链路。先用下面顺序最稳：
1. `PREDICTOR=heuristic` + `CONTEXT_GENERATION_MODE=template` 跑通基础生成。
2. 再打开上下文 LLM。
3. 最后再打开修正 LLM。

这样定位问题最快。

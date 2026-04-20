#!/usr/bin/env bash

# 严格模式：任一错误立刻退出，不做兜底
set -euo pipefail

# 脚本路径与仓库根目录（保证在任意目录执行都能定位到仓库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 自动加载仓库根目录 .env（可选）
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# -----------------------------
# 1) 运行环境配置（按需修改）
# -----------------------------

# 必填：用于画图的 Python 解释器绝对路径
# 示例：/path/to/conda/envs/xxx/bin/python
PLOT_PYTHON="${PLOT_PYTHON:-}"

# 可选：用于生成数据的 Python（默认 python3）
DATA_PYTHON="${DATA_PYTHON:-python3}"

# 可选：matplotlib 缓存目录（建议保留默认）
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_demo}"

# -----------------------------
# 2) 数据与任务配置（按需修改）
# -----------------------------
NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_PLOTS="${NUM_PLOTS:-4}"
SCENE="${SCENE:-temporal_physical_event}"
DATA_TYPE="${DATA_TYPE:-ts_cov_text}"
SEQ_LEN="${SEQ_LEN:-384}"
HISTORY_LEN="${HISTORY_LEN:-192}"
SAMPLING_MINUTES="${SAMPLING_MINUTES:-15}"
SEED="${SEED:-20260402}"

# -----------------------------
# 3) 预测器配置（按需修改）
# -----------------------------
PREDICTOR="${PREDICTOR:-chronos2}"

# 当 PREDICTOR=chronos2 时必填：Chronos2 模型目录绝对路径
# 示例：/path/to/Chronos2
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-}"

# 可选：预测设备，留空则由程序自行决定
# 示例：cpu / cuda:0
PREDICTOR_DEVICE="${PREDICTOR_DEVICE:-}"

# 0=预测器失败直接报错；1=允许回退
ALLOW_PREDICTOR_FALLBACK="${ALLOW_PREDICTOR_FALLBACK:-0}"

# -----------------------------
# 4) 文本上下文 LLM 配置（按需修改）
# -----------------------------
CONTEXT_GENERATION_MODE="${CONTEXT_GENERATION_MODE:-llm}"

# 可选：兼容 OpenAI 协议的 base url，留空则用后端默认值
# 示例：https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_BASE_URL="${LLM_BASE_URL:-}"

# 当 CONTEXT_GENERATION_MODE=llm 时必填：LLM API Key
LLM_API_KEY="${LLM_API_KEY:-}"

# 当 CONTEXT_GENERATION_MODE=llm 时必填：LLM 模型名
# 示例：deepseek-v3.2 / gpt-4o-mini
LLM_MODEL="${LLM_MODEL:-}"

LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-60}"

# 0=LLM 失败直接报错；1=允许回退模板
ALLOW_CONTEXT_FALLBACK="${ALLOW_CONTEXT_FALLBACK:-0}"

# -----------------------------
# 5) 输出配置（按需修改）
# -----------------------------
DATA_OUTPUT="${DATA_OUTPUT:-data/temporal_physical_event_demo_${NUM_SAMPLES}.jsonl}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-data/generate_temporal_reasoning_plots}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-png}"
TITLE_WIDTH="${TITLE_WIDTH:-170}"

# 可选：中文字体文件绝对路径
# 示例：/path/to/SimHei.ttf
FONT_PATH="${FONT_PATH:-}"

# -----------------------------
# 6) 参数校验（缺值就立刻报错）
# -----------------------------
if [[ -z "${PLOT_PYTHON}" ]]; then
  echo "错误：请设置 PLOT_PYTHON（画图 Python 解释器绝对路径）" >&2
  exit 1
fi

if [[ ! -x "${PLOT_PYTHON}" ]]; then
  echo "错误：PLOT_PYTHON 不可执行：${PLOT_PYTHON}" >&2
  exit 1
fi

if [[ "${PREDICTOR}" == "chronos2" && -z "${CHRONOS2_MODEL_PATH}" ]]; then
  echo "错误：PREDICTOR=chronos2 时必须设置 CHRONOS2_MODEL_PATH" >&2
  exit 1
fi

if [[ "${PREDICTOR}" == "chronos2" && ! -d "${CHRONOS2_MODEL_PATH}" ]]; then
  echo "错误：CHRONOS2_MODEL_PATH 目录不存在：${CHRONOS2_MODEL_PATH}" >&2
  exit 1
fi

if [[ "${CONTEXT_GENERATION_MODE}" == "llm" && -z "${LLM_API_KEY}" ]]; then
  echo "错误：CONTEXT_GENERATION_MODE=llm 时必须设置 LLM_API_KEY" >&2
  exit 1
fi

if [[ "${CONTEXT_GENERATION_MODE}" == "llm" && -z "${LLM_MODEL}" ]]; then
  echo "错误：CONTEXT_GENERATION_MODE=llm 时必须设置 LLM_MODEL" >&2
  exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "${DATA_OUTPUT}")" "${PLOT_OUTPUT_DIR}" "${MPLCONFIGDIR}"

# 数据生成命令参数
CLI_ARGS=(
  -m inferential_data_generation.cli
  --scene "${SCENE}"
  --data-type "${DATA_TYPE}"
  --num-samples "${NUM_SAMPLES}"
  --output "${DATA_OUTPUT}"
  --seq-len "${SEQ_LEN}"
  --history-len "${HISTORY_LEN}"
  --sampling-minutes "${SAMPLING_MINUTES}"
  --predictor "${PREDICTOR}"
  --context-generation-mode "${CONTEXT_GENERATION_MODE}"
  --seed "${SEED}"
  --chronos2-model-path "${CHRONOS2_MODEL_PATH}"
  --llm-timeout-seconds "${LLM_TIMEOUT_SECONDS}"
)

# 可视化命令参数
PLOT_ARGS=(
  inferential_data_generation/visualize_demo.py
  --scene "${SCENE}"
  --data-type "${DATA_TYPE}"
  --num-plots "${NUM_PLOTS}"
  --output-dir "${PLOT_OUTPUT_DIR}"
  --output-format "${OUTPUT_FORMAT}"
  --title-width "${TITLE_WIDTH}"
  --seq-len "${SEQ_LEN}"
  --history-len "${HISTORY_LEN}"
  --sampling-minutes "${SAMPLING_MINUTES}"
  --predictor "${PREDICTOR}"
  --context-generation-mode "${CONTEXT_GENERATION_MODE}"
  --seed "${SEED}"
  --chronos2-model-path "${CHRONOS2_MODEL_PATH}"
  --llm-timeout-seconds "${LLM_TIMEOUT_SECONDS}"
)

if [[ -n "${PREDICTOR_DEVICE}" ]]; then
  CLI_ARGS+=(--predictor-device "${PREDICTOR_DEVICE}")
  PLOT_ARGS+=(--predictor-device "${PREDICTOR_DEVICE}")
fi

if [[ "${ALLOW_PREDICTOR_FALLBACK}" != "1" ]]; then
  CLI_ARGS+=(--disable-predictor-fallback)
  PLOT_ARGS+=(--disable-predictor-fallback)
fi

if [[ "${ALLOW_CONTEXT_FALLBACK}" != "1" ]]; then
  CLI_ARGS+=(--disable-context-fallback)
  PLOT_ARGS+=(--disable-context-fallback)
fi

if [[ -n "${LLM_BASE_URL}" ]]; then
  CLI_ARGS+=(--llm-base-url "${LLM_BASE_URL}")
  PLOT_ARGS+=(--llm-base-url "${LLM_BASE_URL}")
fi

if [[ -n "${LLM_API_KEY}" ]]; then
  CLI_ARGS+=(--llm-api-key "${LLM_API_KEY}")
  PLOT_ARGS+=(--llm-api-key "${LLM_API_KEY}")
fi

if [[ -n "${LLM_MODEL}" ]]; then
  CLI_ARGS+=(--llm-model "${LLM_MODEL}")
  PLOT_ARGS+=(--llm-model "${LLM_MODEL}")
fi

if [[ -n "${FONT_PATH}" ]]; then
  PLOT_ARGS+=(--font-path "${FONT_PATH}")
fi

echo "Repository root: ${REPO_ROOT}"
echo "Data python: ${DATA_PYTHON}"
echo "Plot python: ${PLOT_PYTHON}"
echo "Predictor: ${PREDICTOR}"
echo "Scene: ${SCENE}"
echo "Data type: ${DATA_TYPE}"
echo "Context generation mode: ${CONTEXT_GENERATION_MODE}"
echo "Chronos2 model path: ${CHRONOS2_MODEL_PATH}"
echo "Generating dataset to ${DATA_OUTPUT}"
"${DATA_PYTHON}" "${CLI_ARGS[@]}"

echo "Generating demo plots to ${PLOT_OUTPUT_DIR}"
env PYTHONPATH="${REPO_ROOT}" MPLCONFIGDIR="${MPLCONFIGDIR}" \
  "${PLOT_PYTHON}" "${PLOT_ARGS[@]}"

echo "----------------------------------------"
echo "Finished."
echo "Dataset: ${DATA_OUTPUT}"
echo "Plots:   ${PLOT_OUTPUT_DIR}"
echo "----------------------------------------"

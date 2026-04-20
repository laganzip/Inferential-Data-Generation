#!/usr/bin/env bash

# 严格模式：任一错误立刻退出，不做兜底
set -euo pipefail

# 脚本目录与仓库根目录（保证任意位置执行都能定位项目）
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

# 可选：主流程 Python（默认 python3）
DATA_PYTHON="${DATA_PYTHON:-python3}"

# 必填：可视化用 Python 解释器绝对路径
# 示例：/path/to/conda/envs/tsl/bin/python
PLOT_PYTHON="${PLOT_PYTHON:-}"

# 可选：matplotlib 缓存目录
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_demo}"

# -----------------------------
# 2) 运行模式（按需修改）
# -----------------------------
# generate: 只生成+修正
# evaluate: 读取 INPUT_DATA 做修正
# both: 先生成再修正
MODE="${MODE:-evaluate}"  # generate | evaluate | both

# MODE=evaluate 时必填：已有 JSONL 绝对路径
# 示例：/path/to/existing_demo.jsonl
INPUT_DATA="${INPUT_DATA:-}"

# -----------------------------
# 3) 采样参数（按需修改）
# -----------------------------
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEQ_LEN="${SEQ_LEN:-384}"
HISTORY_LEN="${HISTORY_LEN:-192}"
SAMPLING_MINUTES="${SAMPLING_MINUTES:-15}"
SEED="${SEED:-20260409}"

# -----------------------------
# 4) 预测器配置（按需修改）
# -----------------------------
PREDICTOR="${PREDICTOR:-chronos2}"

# 当 PREDICTOR=chronos2 时必填：Chronos2 模型目录绝对路径
# 示例：/path/to/Chronos2
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-}"

# 可选：预测设备（cpu / cuda:0）
PREDICTOR_DEVICE="${PREDICTOR_DEVICE:-}"

# 0=失败直接报错；1=允许回退
ALLOW_PREDICTOR_FALLBACK="${ALLOW_PREDICTOR_FALLBACK:-1}"

# -----------------------------
# 5) 上下文 LLM 配置（按需修改）
# -----------------------------
CONTEXT_GENERATION_MODE="${CONTEXT_GENERATION_MODE:-llm}"

# 当 CONTEXT_GENERATION_MODE=llm 时，以下两项必填
CONTEXT_LLM_API_KEY="${CONTEXT_LLM_API_KEY:-}"
CONTEXT_LLM_MODEL="${CONTEXT_LLM_MODEL:-}"

# 可选：Base URL（兼容 OpenAI 协议）
# 示例：https://open.bigmodel.cn/api/paas/v4
CONTEXT_LLM_BASE_URL="${CONTEXT_LLM_BASE_URL:-}"

CONTEXT_LLM_TIMEOUT_SECONDS="${CONTEXT_LLM_TIMEOUT_SECONDS:-60}"
ALLOW_CONTEXT_FALLBACK="${ALLOW_CONTEXT_FALLBACK:-0}"

# -----------------------------
# 6) 修正 LLM 配置（按需修改）
# -----------------------------
# 当 SKIP_LLM_CORRECTION=0 时，以下三项必填
CORRECTION_LLM_BASE_URL="${CORRECTION_LLM_BASE_URL:-}"
CORRECTION_LLM_API_KEY="${CORRECTION_LLM_API_KEY:-}"
CORRECTION_LLM_MODEL="${CORRECTION_LLM_MODEL:-}"

CORRECTION_LLM_TIMEOUT_SECONDS="${CORRECTION_LLM_TIMEOUT_SECONDS:-360}"
CORRECTION_LLM_MAX_RETRIES="${CORRECTION_LLM_MAX_RETRIES:-2}"
ALLOW_CORRECTION_FALLBACK="${ALLOW_CORRECTION_FALLBACK:-0}"

# 1=跳过修正 LLM（只检查前半流程）
SKIP_LLM_CORRECTION="${SKIP_LLM_CORRECTION:-0}"

# -----------------------------
# 7) 输出配置（按需修改）
# -----------------------------
DATA_OUTPUT="${DATA_OUTPUT:-data/temporal_physical_event_llm_demo_deepseek3_2.jsonl}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-data/test_temporal_reasoning_llm_demo_plots}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-png}"
TITLE_WIDTH="${TITLE_WIDTH:-170}"

# 可选：中文字体绝对路径
FONT_PATH="${FONT_PATH:-}"

# -----------------------------
# 8) 参数校验（缺值立刻报错）
# -----------------------------
if [[ ! "${MODE}" =~ ^(generate|evaluate|both)$ ]]; then
  echo "错误：MODE 只能是 generate / evaluate / both" >&2
  exit 1
fi

if [[ -z "${PLOT_PYTHON}" ]]; then
  echo "错误：请设置 PLOT_PYTHON（可视化 Python 绝对路径）" >&2
  exit 1
fi

if [[ ! -x "${PLOT_PYTHON}" ]]; then
  echo "错误：PLOT_PYTHON 不可执行：${PLOT_PYTHON}" >&2
  exit 1
fi

if [[ "${MODE}" == "evaluate" && -z "${INPUT_DATA}" ]]; then
  echo "错误：MODE=evaluate 时必须设置 INPUT_DATA" >&2
  exit 1
fi

if [[ "${MODE}" == "evaluate" && ! -f "${INPUT_DATA}" ]]; then
  echo "错误：INPUT_DATA 文件不存在：${INPUT_DATA}" >&2
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

if [[ "${CONTEXT_GENERATION_MODE}" == "llm" && -z "${CONTEXT_LLM_API_KEY}" ]]; then
  echo "错误：CONTEXT_GENERATION_MODE=llm 时必须设置 CONTEXT_LLM_API_KEY" >&2
  exit 1
fi

if [[ "${CONTEXT_GENERATION_MODE}" == "llm" && -z "${CONTEXT_LLM_MODEL}" ]]; then
  echo "错误：CONTEXT_GENERATION_MODE=llm 时必须设置 CONTEXT_LLM_MODEL" >&2
  exit 1
fi

if [[ "${SKIP_LLM_CORRECTION}" != "1" ]]; then
  if [[ -z "${CORRECTION_LLM_BASE_URL}" || -z "${CORRECTION_LLM_API_KEY}" || -z "${CORRECTION_LLM_MODEL}" ]]; then
    echo "错误：未跳过修正时，必须设置 CORRECTION_LLM_BASE_URL / CORRECTION_LLM_API_KEY / CORRECTION_LLM_MODEL" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "${DATA_OUTPUT}")" "${PLOT_OUTPUT_DIR}" "${MPLCONFIGDIR}"

# 主流程参数
GEN_ARGS=(
  -m inferential_data_generation.main.generate_llm_correction_demo
  --task random
  --num-samples "${NUM_SAMPLES}"
  --output "${DATA_OUTPUT}"
  --seq-len "${SEQ_LEN}"
  --history-len "${HISTORY_LEN}"
  --sampling-minutes "${SAMPLING_MINUTES}"
  --seed "${SEED}"
  --predictor "${PREDICTOR}"
  --chronos2-model-path "${CHRONOS2_MODEL_PATH}"
  --context-generation-mode "${CONTEXT_GENERATION_MODE}"
  --context-llm-timeout-seconds "${CONTEXT_LLM_TIMEOUT_SECONDS}"
  --correction-llm-timeout-seconds "${CORRECTION_LLM_TIMEOUT_SECONDS}"
  --correction-llm-max-retries "${CORRECTION_LLM_MAX_RETRIES}"
)

# 可视化参数
VIS_ARGS=(
  -m inferential_data_generation.main.visualize_llm_correction_demo
  --input "${DATA_OUTPUT}"
  --output-dir "${PLOT_OUTPUT_DIR}"
  --output-format "${OUTPUT_FORMAT}"
  --title-width "${TITLE_WIDTH}"
  --seq-len "${SEQ_LEN}"
  --history-len "${HISTORY_LEN}"
  --sampling-minutes "${SAMPLING_MINUTES}"
)

if [[ -n "${PREDICTOR_DEVICE}" ]]; then
  GEN_ARGS+=(--predictor-device "${PREDICTOR_DEVICE}")
fi

if [[ "${ALLOW_PREDICTOR_FALLBACK}" != "1" ]]; then
  GEN_ARGS+=(--disable-predictor-fallback)
fi

if [[ "${ALLOW_CONTEXT_FALLBACK}" != "1" ]]; then
  GEN_ARGS+=(--disable-context-fallback)
fi

if [[ -n "${CONTEXT_LLM_BASE_URL}" ]]; then
  GEN_ARGS+=(--context-llm-base-url "${CONTEXT_LLM_BASE_URL}")
fi

if [[ -n "${CONTEXT_LLM_API_KEY}" ]]; then
  GEN_ARGS+=(--context-llm-api-key "${CONTEXT_LLM_API_KEY}")
fi

if [[ -n "${CONTEXT_LLM_MODEL}" ]]; then
  GEN_ARGS+=(--context-llm-model "${CONTEXT_LLM_MODEL}")
fi

if [[ -n "${CORRECTION_LLM_BASE_URL}" ]]; then
  GEN_ARGS+=(--correction-llm-base-url "${CORRECTION_LLM_BASE_URL}")
fi

if [[ -n "${CORRECTION_LLM_API_KEY}" ]]; then
  GEN_ARGS+=(--correction-llm-api-key "${CORRECTION_LLM_API_KEY}")
fi

if [[ -n "${CORRECTION_LLM_MODEL}" ]]; then
  GEN_ARGS+=(--correction-llm-model "${CORRECTION_LLM_MODEL}")
fi

if [[ "${SKIP_LLM_CORRECTION}" == "1" ]]; then
  GEN_ARGS+=(--skip-llm-correction)
fi

if [[ "${ALLOW_CORRECTION_FALLBACK}" != "1" ]]; then
  GEN_ARGS+=(--disable-correction-fallback)
fi

if [[ -n "${FONT_PATH}" ]]; then
  VIS_ARGS+=(--font-path "${FONT_PATH}")
fi

echo "Repository root: ${REPO_ROOT}"
echo "Mode: ${MODE}"
echo "Demo JSONL: ${DATA_OUTPUT}"
echo "Input data: ${INPUT_DATA:-<none>}"

if [[ "${MODE}" == "evaluate" ]]; then
  GEN_ARGS+=(--input-data "${INPUT_DATA}")
fi

"${DATA_PYTHON}" "${GEN_ARGS[@]}"

if [[ "${SKIP_LLM_CORRECTION}" == "1" ]]; then
  echo "Skipping visualization because no LLM correction results were generated."
  exit 0
fi

env PYTHONPATH="${REPO_ROOT}" MPLCONFIGDIR="${MPLCONFIGDIR}" \
  "${PLOT_PYTHON}" "${VIS_ARGS[@]}"

echo "----------------------------------------"
echo "Finished."
echo "Demo JSONL: ${DATA_OUTPUT}"
echo "Plots:      ${PLOT_OUTPUT_DIR}"
echo "----------------------------------------"

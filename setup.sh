#!/usr/bin/env bash
# 支持 Ubuntu 20.04 / 22.04，使用 conda 管理依赖
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="unitree_pick_place"
MENAGERIE_SRC="${MUJOCO_MENAGERIE_SRC:-/tmp/mujoco_menagerie}"
ROBOT_DST="${ROOT}/assets/robots/g1"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
MINICONDA_PREFIX="${MINICONDA_PREFIX:-${HOME}/miniconda3}"

print_conda_install_guide() {
  cat <<EOF

未检测到 conda。本项目依赖 Conda 管理 Python 与 MuJoCo 等库。

【方式一】手动安装 Miniconda（推荐）：

  wget ${MINICONDA_URL} -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p ${MINICONDA_PREFIX}
  ${MINICONDA_PREFIX}/bin/conda init bash
  source ~/.bashrc          # 或重新打开终端
  bash setup.sh

【方式二】由本脚本自动安装 Miniconda 到 ${MINICONDA_PREFIX}：

  bash setup.sh --install-miniconda

也可安装 Anaconda：https://www.anaconda.com/download
文档：https://docs.conda.io/en/latest/miniconda.html

EOF
}

install_miniconda() {
  echo "==> 安装 Miniconda 到 ${MINICONDA_PREFIX}"
  mkdir -p "$(dirname "${MINICONDA_PREFIX}")"
  local installer="/tmp/miniconda-$$.sh"
  if ! command -v wget >/dev/null 2>&1; then
    echo "错误: 需要 wget 下载安装包，请先安装: sudo apt install wget"
    exit 1
  fi
  wget -q "${MINICONDA_URL}" -O "${installer}"
  bash "${installer}" -b -p "${MINICONDA_PREFIX}"
  rm -f "${installer}"
  # shellcheck disable=SC1091
  source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
  echo "    Miniconda 已安装。后续请执行: conda activate ${ENV_NAME}"
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if [ -f "${MINICONDA_PREFIX}/bin/conda" ]; then
    # shellcheck disable=SC1091
    source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
    return 0
  fi
  if [ "${1:-}" = "--install-miniconda" ]; then
    install_miniconda
    return 0
  fi
  echo "错误: 未找到 conda"
  print_conda_install_guide
  exit 1
}

echo "==> 检测系统版本"
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "    ${NAME} ${VERSION_ID}"
  case "${VERSION_ID}" in
    20.04|22.04) ;;
    *)
      echo "警告: 未在 20.04/22.04 上测试，可能需手动调整依赖"
      ;;
  esac
fi

INSTALL_FLAG=""
if [ "${1:-}" = "--install-miniconda" ]; then
  INSTALL_FLAG="--install-miniconda"
fi
ensure_conda ${INSTALL_FLAG}

echo "==> 创建/更新 conda 环境: ${ENV_NAME}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ROOT}/environment.yml" --prune
else
  conda env create -f "${ROOT}/environment.yml"
fi

echo "==> 安装宇树 G1 机器人模型 (MuJoCo Menagerie)"
mkdir -p "${ROOT}/assets/robots"
if [ -d "${MENAGERIE_SRC}/unitree_g1" ]; then
  rm -rf "${ROBOT_DST}"
  cp -a "${MENAGERIE_SRC}/unitree_g1" "${ROBOT_DST}"
  echo "    已从 ${MENAGERIE_SRC} 复制 G1 模型"
else
  echo "    本地未找到模型，从 Gitee 克隆 mujoco_menagerie ..."
  TMP_CLONE="${ROOT}/third_party/mujoco_menagerie"
  rm -rf "${TMP_CLONE}"
  git clone --depth 1 https://gitee.com/liu_sh/mujoco_menagerie.git "${TMP_CLONE}"
  cp -a "${TMP_CLONE}/unitree_g1" "${ROBOT_DST}"
fi

SCENE_SRC="${ROOT}/assets/scenes/box_pick_place.xml"
if [ -f "${SCENE_SRC}" ]; then
  cp "${SCENE_SRC}" "${ROBOT_DST}/box_pick_place.xml"
  echo "    已部署自定义场景 box_pick_place.xml"
else
  echo "警告: 未找到 ${SCENE_SRC}，验证步骤可能失败"
fi

echo "==> 下载 G1 RL 行走策略 (unitree_rl_gym)"
bash "${ROOT}/scripts/download_policy.sh"

echo "==> 验证 MuJoCo、策略与场景"
conda run -n "${ENV_NAME}" python -c "
import mujoco
import torch
from pathlib import Path
from src.control.locomotion import G1LocomotionPolicy
from src.control.walker import PolicyWalker, LEG_ACTUATOR_NAMES

p = Path('${ROOT}') / 'assets/robots/g1/box_pick_place.xml'
m = mujoco.MjModel.from_xml_path(str(p))
leg_ids = [m.actuator(n).id for n in LEG_ACTUATOR_NAMES]
loco = G1LocomotionPolicy.from_yaml(m, leg_ids, LEG_ACTUATOR_NAMES)
print('MuJoCo', mujoco.__version__, '| Torch', torch.__version__)
print('G1 场景 nq=', m.nq, 'nu=', m.nu, '| RL 行走策略已加载')
"

echo ""
echo "环境就绪。运行仿真："
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/run_sim.py"
echo "  python scripts/run_sim.py --demo   # 自动演示搬运"

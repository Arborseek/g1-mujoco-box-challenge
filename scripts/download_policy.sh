#!/usr/bin/env bash
# 下载宇树 unitree_rl_gym G1 行走策略
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/g1_policy/motion.pt"
mkdir -p "$(dirname "${DEST}")"

if [ -f "${DEST}" ] && [ "$(stat -c%s "${DEST}")" -gt 100000 ]; then
  echo "策略已存在: ${DEST}"
  exit 0
fi

URLS=(
  "https://ghproxy.net/https://github.com/unitreerobotics/unitree_rl_gym/raw/main/deploy/pre_train/g1/motion.pt"
  "https://github.com/unitreerobotics/unitree_rl_gym/raw/main/deploy/pre_train/g1/motion.pt"
)

for url in "${URLS[@]}"; do
  echo "尝试下载: ${url}"
  if curl -L --retry 2 --connect-timeout 15 -o "${DEST}.tmp" "${url}" && \
     [ "$(stat -c%s "${DEST}.tmp")" -gt 100000 ]; then
    mv "${DEST}.tmp" "${DEST}"
    echo "下载成功: ${DEST} ($(stat -c%s "${DEST}") bytes)"
    exit 0
  fi
  rm -f "${DEST}.tmp"
done

echo "错误: 策略下载失败，请手动下载 motion.pt 到 ${DEST}"
exit 1

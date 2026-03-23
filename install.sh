#!/bin/bash
# Intel XPU 환경 패키지 설치 스크립트
#
# +xpu 빌드는 아래 두 인덱스에서 제공됩니다:
#   - https://download.pytorch.org/whl/xpu         (torch, torchvision, torchaudio, pytorch-triton-xpu)
#   - https://pytorch-extension.intel.com/...       (intel_extension_for_pytorch)
# 일반 pip install -r requirements.txt 로는 설치되지 않으므로 이 스크립트를 사용하세요.

set -e

pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
  --extra-index-url https://pypi.org/simple/

echo "설치 완료!"

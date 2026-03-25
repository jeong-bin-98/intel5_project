# 📊 Inference Benchmark: CPU vs OpenVINO vs XPU

YOLO11n-OBB 모델의 추론 속도를 3가지 디바이스/런타임 설정에서 **독립적으로** 측정하고 비교합니다.

## 테스트 대상

| 설정 | 설명 | 스크립트 |
|---|---|---|
| **CPU (PyTorch)** | OpenVINO 미사용, PyTorch 네이티브 추론 | `test_benchmark_cpu.py` |
| **CPU (OpenVINO)** | `.pt` → OpenVINO IR 자동 변환 후 추론 | `test_benchmark_openvino.py` |
| **GPU (Intel XPU)** | Intel Arc A770 XPU 네이티브 추론 | `test_benchmark_gpu.py` |

## 사용법 — Inference-only 벤치마크

### 1단계: 개별 벤치마크 실행

각 스크립트를 **별도로** 실행하여 메모리/캐시 간섭 없이 정밀 측정합니다.

```bash
# 프로젝트 루트에서 실행
cd /home/intel5/Documents/JB/Intel5

# CPU (PyTorch) 벤치마크
python src/detection/tests/benchmark/test_benchmark_cpu.py --runs 50

# CPU (OpenVINO) 벤치마크 — 최초 실행 시 자동으로 OpenVINO IR 변환
python src/detection/tests/benchmark/test_benchmark_openvino.py --runs 50

# GPU (Intel XPU) 벤치마크
python src/detection/tests/benchmark/test_benchmark_gpu.py --runs 50
```

> ⚠️ **정밀 측정을 위해 각 스크립트를 따로따로 실행하세요!**
> 한 번에 모두 돌리면 메모리 잔류/캐시 영향으로 정확한 비교가 어렵습니다.

### 2단계: 결과 비교

```bash
python src/detection/tests/benchmark/test_benchmark_compare.py
```

## 사용법 — Full Pipeline 벤치마크

YOLO 추론 + Depth 처리 + RANSAC 법선 추정 + Approach 벡터 계산까지 전체 `detect_3d()` 파이프라인을 측정합니다.

### 1단계: 테스트 프레임 캡처 (최초 1회, 카메라 필요)

```bash
python src/detection/tests/benchmark/test_benchmark_pipeline.py --capture
```

### 2단계: 파이프라인 벤치마크 실행

```bash
python src/detection/tests/benchmark/test_benchmark_pipeline.py --runs 30
```

### 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--runs N` | 반복 추론 횟수 | 50 (pipeline: 30) |
| `--warmup N` | 워밍업 횟수 | 10 (pipeline: 5) |
| `--img PATH` | 특정 이미지로 테스트 (inference-only) | val 폴더 첫 번째 이미지 |
| `--capture` | (pipeline 전용) RealSense 테스트 프레임 캡처 | - |

## 결과 파일

```
runs/benchmark/
├── result_cpu.json               # CPU 벤치마크 결과
├── result_openvino.json          # OpenVINO 벤치마크 결과
├── result_gpu.json               # XPU 벤치마크 결과
├── result_pipeline_cpu.json      # CPU 파이프라인 결과
├── result_pipeline_openvino.json # OpenVINO 파이프라인 결과
├── result_pipeline_gpu.json      # XPU 파이프라인 결과
├── comparison.csv                # Inference-only 비교 테이블
├── comparison.png                # Inference-only 비교 그래프
├── comparison_full.csv           # Inference + Pipeline 통합 CSV
├── comparison_pipeline.png       # Inference vs Pipeline 비교 그래프
└── test_frame/                   # 파이프라인용 저장 테스트 프레임
    ├── color.npy
    ├── depth.npy
    └── intrinsics.json
```

## 측정 원칙

- **독립 프로세스**: 각 벤치마크를 별도 Python 프로세스로 실행하여 메모리/캐시 간섭 방지
- **Warm-up**: 본 측정 전 10회 추론으로 JIT 컴파일 및 캐시 안정화
- **XPU Sync**: XPU 측정 시 `torch.xpu.synchronize()` 사용으로 비동기 연산 완료 대기
- **동일 이미지**: 모든 설정에서 동일한 이미지로 테스트하여 공정한 비교
- **자동 변환**: OpenVINO 벤치마크 최초 실행 시 `.pt` → OpenVINO IR 변환 자동 처리

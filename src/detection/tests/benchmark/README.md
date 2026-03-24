# 📊 Inference Benchmark: CPU vs OpenVINO vs GPU

YOLO11n-OBB 모델의 추론 속도를 3가지 디바이스/런타임 설정에서 **독립적으로** 측정하고 비교합니다.

## 테스트 대상

| 설정 | 설명 | 스크립트 |
|---|---|---|
| **CPU (PyTorch)** | OpenVINO 미사용, PyTorch 네이티브 추론 | `test_benchmark_cpu.py` |
| **CPU (OpenVINO)** | `.pt` → OpenVINO IR 변환 후 추론 | `test_benchmark_openvino.py` |
| **GPU (CUDA)** | GTX 1650 PyTorch 네이티브 추론 | `test_benchmark_gpu.py` |

## 사용법

### 1단계: OpenVINO 모델 변환 (최초 1회)

```bash
cd c:\Users\hp\Documents\Intel5
python src/detection/tests/benchmark/convert_openvino.py
```

### 2단계: 개별 벤치마크 실행

각 스크립트를 **별도로** 실행하여 메모리/캐시 간섭 없이 정밀 측정합니다.

```bash
# CPU (PyTorch) 벤치마크
python src/detection/tests/benchmark/test_benchmark_cpu.py --runs 50

# CPU (OpenVINO) 벤치마크 — 순수 OpenVINO API 사용
python src/detection/tests/benchmark/test_benchmark_openvino.py --runs 50

# GPU (CUDA) 벤치마크
python src/detection/tests/benchmark/test_benchmark_gpu.py --runs 50
```

> ⚠️ **정밀 측정을 위해 각 스크립트를 따로따로 실행하세요!**
> 한 번에 모두 돌리면 메모리 잔류/캐시 영향으로 정확한 비교가 어렵습니다.

### 3단계: 결과 비교

```bash
python src/detection/tests/benchmark/test_benchmark_compare.py
```

### 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--runs N` | 반복 추론 횟수 | 50 |
| `--warmup N` | 워밍업 횟수 | 10 |
| `--img PATH` | 특정 이미지로 테스트 | val 폴더 첫 번째 이미지 |

## 결과 파일

```
runs/benchmark/
├── result_cpu.json          # CPU 벤치마크 결과
├── result_openvino.json     # OpenVINO 벤치마크 결과
├── result_gpu.json          # GPU 벤치마크 결과
├── comparison.csv           # 비교 테이블 (compare 스크립트 생성)
└── comparison.png           # 비교 막대 그래프 (compare 스크립트 생성)
```

## 측정 원칙

- **독립 프로세스**: 각 벤치마크를 별도 Python 프로세스로 실행하여 메모리/캐시 간섭 방지
- **Warm-up**: 본 측정 전 10회 추론으로 JIT 컴파일 및 캐시 안정화
- **CUDA Sync**: GPU 측정 시 `torch.cuda.synchronize()` 사용으로 비동기 연산 완료 대기
- **동일 이미지**: 모든 설정에서 동일한 이미지로 테스트하여 공정한 비교

"""
CPU (OpenVINO) Inference 벤치마크
=================================

순수 OpenVINO API (ov.Core → compile_model → infer_request.infer)를 사용하여
CPU 추론 속도를 측정합니다. ultralytics 래퍼 오버헤드 없이 순수 OpenVINO 성능만 측정합니다.

사전 준비:
    python src/detection/tests/benchmark/convert_openvino.py

사용법:
    python src/detection/tests/benchmark/test_benchmark_openvino.py --runs 50
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import argparse
import json
import time
import glob
import numpy as np
import cv2
from datetime import datetime

import openvino as ov

from config.paths import BENCHMARK_DIR, RUNS_DIR
from src.detection.tests.benchmark.benchmark_utils import find_model, find_test_image


def export_openvino_model():
    """YOLO 모델을 OpenVINO IR로 변환합니다 (이미 있으면 스킵).

    Returns:
        str: OpenVINO .xml 파일 경로
    """
    model_path = find_model()
    model_dir = os.path.dirname(model_path)
    openvino_dir = os.path.join(model_dir, "best_openvino_model")

    # 이미 변환된 모델 확인
    if os.path.isdir(openvino_dir):
        xml_files = glob.glob(os.path.join(openvino_dir, "*.xml"))
        if xml_files:
            print(f"✅ OpenVINO 모델 이미 존재: {openvino_dir}")
            return xml_files[0]

    # ultralytics export로 변환
    print("🔄 OpenVINO IR 변환 중...")
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.export(format="openvino")
    print(f"✅ 변환 완료: {openvino_dir}")

    xml_files = glob.glob(os.path.join(openvino_dir, "*.xml"))
    if xml_files:
        return xml_files[0]

    print("❌ 변환 후 .xml 파일을 찾을 수 없습니다.")
    sys.exit(1)


def preprocess_image(image, input_shape):
    """YOLO 입력에 맞게 이미지를 전처리합니다.

    Args:
        image: BGR 이미지 (H, W, 3)
        input_shape: 모델 입력 shape (N, C, H, W)

    Returns:
        np.ndarray: 전처리된 입력 텐서
    """
    _, _, h, w = input_shape

    # Resize (letterbox 없이 단순 resize — 벤치마크용)
    resized = cv2.resize(image, (w, h))

    # BGR → RGB, HWC → CHW, normalize 0~1
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    # 배치 차원 추가 (1, C, H, W)
    return np.expand_dims(chw, axis=0)


def run_benchmark(runs=50, warmup=10, img_path=None):
    """CPU (OpenVINO) 벤치마크를 실행합니다 — 순수 OpenVINO API 사용."""

    # 이미지 준비
    test_img = find_test_image(img_path)
    image = cv2.imread(test_img)
    print(f"📷 테스트 이미지: {os.path.basename(test_img)} ({image.shape[1]}x{image.shape[0]})")

    # OpenVINO 모델 로드
    xml_path = export_openvino_model()
    print(f"🔧 OpenVINO 모델: {xml_path}")
    print(f"🖥️  디바이스: CPU (OpenVINO - 순수 API)")

    core = ov.Core()
    model = core.read_model(xml_path)
    compiled_model = core.compile_model(model, device_name="CPU", config={
        "PERFORMANCE_HINT": "LATENCY",          # 단일 추론 지연시간 최적화
    })

    # 입력 shape 확인 및 전처리
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape
    print(f"   입력 shape: {input_shape}")

    input_data = preprocess_image(image, input_shape)

    # InferRequest 생성
    infer_request = compiled_model.create_infer_request()

    # Warm-up
    print(f"\n⏳ Warm-up ({warmup}회)...")
    for i in range(warmup):
        infer_request.infer([input_data])
    print("   Warm-up 완료")

    # 본 측정
    print(f"\n🏃 벤치마크 실행 ({runs}회)...")
    times = []
    for i in range(runs):
        start = time.perf_counter()
        infer_request.infer([input_data])
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{runs} 완료 (현재 평균: {np.mean(times):.1f}ms)")

    # 결과 계산
    times_arr = np.array(times)
    result = {
        "device": "CPU (OpenVINO)",
        "model": os.path.basename(xml_path),
        "original_model": os.path.basename(find_model()),
        "image": os.path.basename(test_img),
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "input_shape": str(input_shape),
        "runs": runs,
        "warmup": warmup,
        "avg_ms": round(float(np.mean(times_arr)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr)), 2),
        "median_ms": round(float(np.median(times_arr)), 2),
        "times_ms": [round(t, 2) for t in times],
        "timestamp": datetime.now().isoformat(),
    }

    # 결과 출력
    print("\n" + "=" * 50)
    print("📊 CPU (OpenVINO) 벤치마크 결과")
    print("=" * 50)
    print(f"  평균: {result['avg_ms']:.2f} ms")
    print(f"  중앙: {result['median_ms']:.2f} ms")
    print(f"  최소: {result['min_ms']:.2f} ms")
    print(f"  최대: {result['max_ms']:.2f} ms")
    print(f"  표준편차: {result['std_ms']:.2f} ms")
    print("=" * 50)

    # JSON 저장
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    out_path = os.path.join(BENCHMARK_DIR, "result_openvino.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장: {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU (OpenVINO) Inference Benchmark")
    parser.add_argument("--runs", type=int, default=50, help="반복 추론 횟수 (기본: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="워밍업 횟수 (기본: 10)")
    parser.add_argument("--img", type=str, default=None, help="테스트 이미지 경로")
    args = parser.parse_args()

    run_benchmark(runs=args.runs, warmup=args.warmup, img_path=args.img)

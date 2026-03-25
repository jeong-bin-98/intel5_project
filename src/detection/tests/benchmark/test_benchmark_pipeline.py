"""
Full Pipeline Benchmark (detect_3d)
====================================

YOLO 추론 + depth 처리 + RANSAC 법선 추정 + approach 벡터 계산까지
전체 detect_3d() 파이프라인을 CPU, OpenVINO, XPU로 측정합니다.

기존 inference-only 벤치마크와 달리, 실제 binpicking에서의
end-to-end 지연시간을 측정합니다.

사전 준비:
    RealSense 카메라로 테스트 프레임 저장 (최초 1회):
    python src/detection/tests/benchmark/test_benchmark_pipeline.py --capture

사용법:
    python src/detection/tests/benchmark/test_benchmark_pipeline.py
    python src/detection/tests/benchmark/test_benchmark_pipeline.py --runs 50 --warmup 10
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import argparse
import json
import time
import glob
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO

from config.paths import BENCHMARK_DIR
from src.detection.constants import DEVICE, CONFIDENCE
from src.detection.binpicking_3d import detect_3d
from src.detection.tests.benchmark.benchmark_utils import (
    find_model, capture_test_frame, load_test_frame, has_test_frame,
)


def export_openvino_model():
    """YOLO 모델을 OpenVINO IR로 변환합니다 (이미 있으면 스킵).

    Returns:
        str: OpenVINO 모델 디렉토리 경로
    """
    model_path = find_model()
    model_dir = os.path.dirname(model_path)
    openvino_dir = os.path.join(model_dir, "best_openvino_model")

    if os.path.isdir(openvino_dir):
        xml_files = glob.glob(os.path.join(openvino_dir, "*.xml"))
        if xml_files:
            print(f"   OpenVINO 모델 존재: {openvino_dir}")
            return openvino_dir

    print("   OpenVINO IR 변환 중...")
    model = YOLO(model_path)
    model.export(format="openvino")
    print(f"   변환 완료: {openvino_dir}")
    return openvino_dir


def benchmark_one(device_name, model, color_image, depth_image, intrinsics,
                  device_str, runs=30, warmup=5):
    """특정 디바이스로 detect_3d() 파이프라인을 벤치마크합니다.

    Args:
        device_name: 결과에 표시할 디바이스 이름
        model: YOLO 모델 인스턴스
        color_image: 컬러 이미지 (H, W, 3)
        depth_image: depth 이미지 (H, W) - mm
        intrinsics: RealSense intrinsics
        device_str: YOLO에 전달할 device 문자열
        runs: 벤치마크 반복 횟수
        warmup: 워밍업 횟수

    Returns:
        dict: 벤치마크 결과
    """
    is_xpu = device_str.startswith("xpu") and hasattr(torch, 'xpu')

    print(f"\n{'='*55}")
    print(f"  {device_name} — Full Pipeline")
    print(f"{'='*55}")

    # Warm-up
    print(f"  Warm-up ({warmup}회)...")
    for _ in range(warmup):
        detect_3d(model, color_image, depth_image, intrinsics, device=device_str)
        if is_xpu:
            torch.xpu.synchronize()
    print("  Warm-up 완료")

    # 본 측정
    print(f"  벤치마크 실행 ({runs}회)...")
    times = []
    det_counts = []

    for i in range(runs):
        if is_xpu:
            torch.xpu.synchronize()

        start = time.perf_counter()
        objects = detect_3d(model, color_image, depth_image, intrinsics,
                           device=device_str)
        if is_xpu:
            torch.xpu.synchronize()

        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
        det_counts.append(len(objects))

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{runs} 완료 "
                  f"(avg: {np.mean(times):.1f}ms, det: {len(objects)}개)")

    times_arr = np.array(times)
    result = {
        "device": device_name,
        "benchmark_type": "full_pipeline",
        "runs": runs,
        "warmup": warmup,
        "avg_ms": round(float(np.mean(times_arr)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr)), 2),
        "median_ms": round(float(np.median(times_arr)), 2),
        "avg_detections": round(float(np.mean(det_counts)), 1),
        "times_ms": [round(t, 2) for t in times],
        "timestamp": datetime.now().isoformat(),
    }

    # 결과 요약
    print(f"\n  {'평균':>6s}: {result['avg_ms']:>8.2f} ms")
    print(f"  {'중앙':>6s}: {result['median_ms']:>8.2f} ms")
    print(f"  {'최소':>6s}: {result['min_ms']:>8.2f} ms")
    print(f"  {'최대':>6s}: {result['max_ms']:>8.2f} ms")
    print(f"  {'표준편차':>6s}: {result['std_ms']:>8.2f} ms")
    print(f"  {'탐지수':>6s}: {result['avg_detections']:>8.1f} 개/프레임")

    return result


def run_pipeline_benchmark(runs=30, warmup=5):
    """전체 파이프라인 벤치마크를 실행합니다 (CPU, OpenVINO, XPU)."""

    print("=" * 55)
    print("  Full Pipeline Benchmark (detect_3d)")
    print("  YOLO + Depth + RANSAC + Approach Vector")
    print("=" * 55)

    # 1) 테스트 프레임 로드
    color_image, depth_image, intrinsics = load_test_frame()

    model_path = find_model()
    print(f"   모델: {os.path.basename(model_path)}")
    print(f"   이미지: {color_image.shape[1]}x{color_image.shape[0]}")

    all_results = []

    # ── CPU (PyTorch) ──
    print(f"\n모델 로드: {os.path.basename(model_path)} → CPU")
    model_cpu = YOLO(model_path)
    result_cpu = benchmark_one(
        "CPU (PyTorch)", model_cpu, color_image, depth_image, intrinsics,
        device_str="cpu", runs=runs, warmup=warmup,
    )
    all_results.append(("pipeline_cpu", result_cpu))
    del model_cpu

    # ── CPU (OpenVINO via Ultralytics) ──
    try:
        ov_dir = export_openvino_model()
        print(f"\n모델 로드: OpenVINO → CPU")
        model_ov = YOLO(ov_dir, task="obb")
        result_ov = benchmark_one(
            "CPU (OpenVINO)", model_ov, color_image, depth_image, intrinsics,
            device_str="auto", runs=runs, warmup=warmup,
        )
        all_results.append(("pipeline_openvino", result_ov))
        del model_ov
    except Exception as e:
        print(f"\n⚠️  OpenVINO 벤치마크 스킵: {e}")

    # ── XPU (Intel Arc) ──
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        gpu_name = torch.xpu.get_device_name(0)
        print(f"\n모델 로드: {os.path.basename(model_path)} → XPU ({gpu_name})")
        model_xpu = YOLO(model_path)
        model_xpu.to("xpu")
        result_xpu = benchmark_one(
            f"XPU ({gpu_name})", model_xpu, color_image, depth_image, intrinsics,
            device_str=DEVICE, runs=runs, warmup=warmup,
        )
        all_results.append(("pipeline_gpu", result_xpu))
        del model_xpu
    else:
        print("\n⚠️  XPU 사용 불가 — 스킵")

    # ── 결과 저장 ──
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    for filename, result in all_results:
        out_path = os.path.join(BENCHMARK_DIR, f"result_{filename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 저장: {out_path}")

    # ── 비교 테이블 ──
    if len(all_results) >= 2:
        print_comparison([r for _, r in all_results])

    print("\n✅ 파이프라인 벤치마크 완료!")


def print_comparison(results):
    """파이프라인 벤치마크 결과를 비교 출력합니다."""
    print("\n" + "=" * 70)
    print("  Full Pipeline 비교 결과")
    print("=" * 70)
    print(f"  {'설정':<25s} {'평균(ms)':>10s} {'중앙(ms)':>10s} "
          f"{'최소(ms)':>10s} {'탐지수':>8s}")
    print("  " + "-" * 63)

    for r in results:
        name = r["device"]
        if len(name) > 25:
            name = name[:22] + "..."
        print(f"  {name:<25s} {r['avg_ms']:>10.2f} {r['median_ms']:>10.2f} "
              f"{r['min_ms']:>10.2f} {r['avg_detections']:>7.1f}개")

    print("=" * 70)

    # Speedup 요약
    fastest = min(results, key=lambda r: r["avg_ms"])
    slowest = max(results, key=lambda r: r["avg_ms"])
    ratio = slowest["avg_ms"] / fastest["avg_ms"]

    print(f"\n  가장 빠름: {fastest['device']} — {fastest['avg_ms']:.2f}ms "
          f"({1000/fastest['avg_ms']:.1f} FPS)")
    print(f"  가장 느림: {slowest['device']} — {slowest['avg_ms']:.2f}ms "
          f"({1000/slowest['avg_ms']:.1f} FPS)")
    print(f"  최대 차이: {ratio:.2f}배")

    # OpenVINO vs CPU
    cpu_r = next((r for r in results if "PyTorch" in r["device"]), None)
    ov_r = next((r for r in results if "OpenVINO" in r["device"]), None)
    if cpu_r and ov_r:
        sp = cpu_r["avg_ms"] / ov_r["avg_ms"]
        print(f"  OpenVINO 효과: CPU 대비 {sp:.2f}배 속도 향상 (파이프라인 전체)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Pipeline Benchmark (detect_3d)")
    parser.add_argument("--capture", action="store_true",
                        help="RealSense에서 테스트 프레임 캡처 후 저장")
    parser.add_argument("--runs", type=int, default=30,
                        help="반복 횟수 (기본: 30)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="워밍업 횟수 (기본: 5)")
    args = parser.parse_args()

    if args.capture:
        capture_test_frame()
        print("\n테스트 프레임 저장 완료. 벤치마크를 실행하려면:")
        print("  python src/detection/tests/benchmark/test_benchmark_pipeline.py")
    else:
        run_pipeline_benchmark(runs=args.runs, warmup=args.warmup)

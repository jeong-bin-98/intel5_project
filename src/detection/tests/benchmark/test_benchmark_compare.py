"""
벤치마크 결과 비교 분석
========================

개별 벤치마크(CPU, OpenVINO, GPU)의 JSON 결과를 모아서
비교 테이블, Speedup Matrix, 막대 그래프를 생성합니다.

사전 준비:
    아래 스크립트를 각각 **따로** 실행하세요:
    - test_benchmark_cpu.py
    - test_benchmark_openvino.py
    - test_benchmark_gpu.py  (Intel XPU)

사용법:
    python src/detection/tests/benchmark/test_benchmark_compare.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import json
import csv
from config.paths import BENCHMARK_DIR


# 결과 파일 이름과 표시 순서
RESULT_FILES = [
    ("result_cpu.json", "CPU (PyTorch)"),
    ("result_openvino.json", "CPU (OpenVINO)"),
    ("result_gpu.json", "XPU (Intel Arc)"),
]

# 파이프라인 벤치마크 결과 (detect_3d 전체 — 저장 프레임 기반)
PIPELINE_RESULT_FILES = [
    ("result_pipeline_cpu.json", "CPU (PyTorch)"),
    ("result_pipeline_openvino.json", "CPU (OpenVINO)"),
    ("result_pipeline_gpu.json", "XPU (Intel Arc)"),
]

# 라이브 파이프라인 벤치마크 결과 (binpicking_3d.py 실시간 측정)
LIVE_RESULT_FILES = [
    ("live_cpu.json", "CPU (PyTorch)"),
    ("live_openvino.json", "CPU (OpenVINO)"),
    ("live_xpu.json", "XPU (Intel Arc)"),
]


def load_results():
    """저장된 벤치마크 결과를 로드합니다."""
    results = []
    missing = []

    for filename, display_name in RESULT_FILES:
        path = os.path.join(BENCHMARK_DIR, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_display_name"] = display_name
            data["_filename"] = filename
            results.append(data)
        else:
            missing.append((filename, display_name))

    return results, missing


def print_individual_table(results):
    """개별 성능 결과 테이블을 출력합니다."""
    print("\n" + "=" * 70)
    print("  📊 개별 성능 결과")
    print("=" * 70)
    print(f"  {'설정':<20s} {'평균(ms)':>10s} {'중앙(ms)':>10s} {'최소(ms)':>10s} {'최대(ms)':>10s}")
    print("  " + "-" * 58)

    for r in results:
        name = r["_display_name"]
        print(f"  {name:<20s} {r['avg_ms']:>10.2f} {r['median_ms']:>10.2f} "
              f"{r['min_ms']:>10.2f} {r['max_ms']:>10.2f}")

    print("=" * 70)


def print_speedup_matrix(results):
    """Speedup Matrix (쌍별 속도 비율)를 출력합니다."""
    n = len(results)

    # 헤더 준비
    short_names = []
    for r in results:
        name = r["_display_name"]
        # 짧은 이름
        if "OpenVINO" in name:
            short_names.append("OpenVINO")
        elif "XPU" in name:
            short_names.append("XPU")
        else:
            short_names.append("CPU(PT)")

    print("\n" + "=" * 70)
    print("  🔄 속도 비교 (Speedup Matrix)")
    print("  읽는 법: '행'이 기준, '열' 대비 몇 배 빠른지")
    print("=" * 70)

    # 헤더
    header = f"  {'기준 \\ 대상':<16s}"
    for sn in short_names:
        header += f"{sn:>12s}"
    print(header)
    print("  " + "-" * (14 + 12 * n))

    # 행 출력
    for i, r_base in enumerate(results):
        row = f"  {short_names[i]:<16s}"
        for j, r_target in enumerate(results):
            ratio = r_base["avg_ms"] / r_target["avg_ms"]
            row += f"{ratio:>11.2f}x"
        print(row)

    print("=" * 70)

    # 요약
    fastest = min(results, key=lambda r: r["avg_ms"])
    slowest = max(results, key=lambda r: r["avg_ms"])
    print(f"\n  ✅ 가장 빠른 설정: {fastest['_display_name']} — {fastest['avg_ms']:.2f}ms/frame")
    print(f"  🐢 가장 느린 설정: {slowest['_display_name']} — {slowest['avg_ms']:.2f}ms/frame")
    print(f"  📈 최대 속도 차이: {slowest['avg_ms'] / fastest['avg_ms']:.2f}배")

    # OpenVINO 효과 (CPU vs OpenVINO 비교)
    cpu_result = next((r for r in results if "PyTorch" in r["_display_name"]), None)
    ov_result = next((r for r in results if "OpenVINO" in r["_display_name"]), None)
    if cpu_result and ov_result:
        ov_speedup = cpu_result["avg_ms"] / ov_result["avg_ms"]
        print(f"\n  💡 OpenVINO 효과: CPU(PyTorch) 대비 {ov_speedup:.2f}배 속도 향상")

    print()


def save_csv(results):
    """비교 결과를 CSV로 저장합니다."""
    csv_path = os.path.join(BENCHMARK_DIR, "comparison.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["설정", "평균(ms)", "중앙값(ms)", "최소(ms)", "최대(ms)",
                         "표준편차(ms)", "반복횟수", "측정시각"])
        for r in results:
            writer.writerow([
                r["_display_name"], r["avg_ms"], r["median_ms"],
                r["min_ms"], r["max_ms"], r["std_ms"],
                r["runs"], r.get("timestamp", "")
            ])

    print(f"  📄 CSV 저장: {csv_path}")
    return csv_path


def save_chart(results):
    """비교 막대 그래프를 생성합니다."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # GUI 없이 저장
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ matplotlib가 설치되지 않아 그래프를 생성할 수 없습니다.")
        return None

    names = [r["_display_name"] for r in results]
    avg_times = [r["avg_ms"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 색상 설정
    colors = ["#4A90D9", "#50C878", "#FF6B6B"][:len(results)]

    # 평균 추론 시간 막대 그래프
    bars = ax.bar(names, avg_times, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Avg Inference Latency (ms)", fontsize=12)
    ax.set_title("Inference Latency Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 값 표시
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()

    chart_path = os.path.join(BENCHMARK_DIR, "comparison.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 그래프 저장: {chart_path}")
    return chart_path


def load_pipeline_results():
    """파이프라인 벤치마크 결과를 로드합니다."""
    results = []
    missing = []

    for filename, display_name in PIPELINE_RESULT_FILES:
        path = os.path.join(BENCHMARK_DIR, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_display_name"] = display_name
            data["_filename"] = filename
            results.append(data)
        else:
            missing.append((filename, display_name))

    return results, missing


def load_live_results():
    """라이브 파이프라인 벤치마크 결과를 로드합니다."""
    results = []
    missing = []

    for filename, display_name in LIVE_RESULT_FILES:
        path = os.path.join(BENCHMARK_DIR, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_display_name"] = display_name
            data["_filename"] = filename
            # live 결과에는 runs가 없을 수 있으므로 measured_frames 사용
            if "runs" not in data:
                data["runs"] = data.get("measured_frames", 0)
            results.append(data)
        else:
            missing.append((filename, display_name))

    return results, missing


def print_pipeline_table(results, title="Full Pipeline 성능 결과 (detect_3d 전체)"):
    """파이프라인 벤치마크 결과 테이블을 출력합니다."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("     YOLO + Depth + RANSAC + Approach Vector")
    print("=" * 70)
    print(f"  {'설정':<20s} {'평균(ms)':>10s} {'중앙(ms)':>10s} "
          f"{'최소(ms)':>10s} {'최대(ms)':>10s} {'FPS':>8s}")
    print("  " + "-" * 66)

    for r in results:
        name = r["_display_name"]
        fps = r.get("fps", 1000.0 / r["avg_ms"] if r["avg_ms"] > 0 else 0)
        print(f"  {name:<20s} {r['avg_ms']:>10.2f} {r['median_ms']:>10.2f} "
              f"{r['min_ms']:>10.2f} {r['max_ms']:>10.2f} {fps:>7.1f}")

    print("=" * 70)


def print_combined_comparison(infer_results, pipeline_results):
    """Inference-only vs Full Pipeline 비교 테이블을 출력합니다."""
    print("\n" + "=" * 70)
    print("  🔍 Inference-only vs Full Pipeline 비교")
    print("=" * 70)
    print(f"  {'설정':<20s} {'추론(ms)':>10s} {'파이프(ms)':>10s} "
          f"{'후처리(ms)':>12s} {'추론 비율':>10s}")
    print("  " + "-" * 60)

    for ir in infer_results:
        name = ir["_display_name"]
        # 같은 설정의 파이프라인 결과 찾기
        pr = next((p for p in pipeline_results
                   if p["_display_name"] == name), None)
        if pr:
            overhead = pr["avg_ms"] - ir["avg_ms"]
            infer_pct = ir["avg_ms"] / pr["avg_ms"] * 100
            print(f"  {name:<20s} {ir['avg_ms']:>10.2f} {pr['avg_ms']:>10.2f} "
                  f"{overhead:>10.2f}ms {infer_pct:>9.1f}%")
        else:
            print(f"  {name:<20s} {ir['avg_ms']:>10.2f} {'(없음)':>10s} "
                  f"{'—':>12s} {'—':>10s}")

    print("=" * 70)
    print("  * 후처리 = Depth 샘플링 + RANSAC 법선 추정 + Approach 벡터 계산")


def save_combined_chart(infer_results, pipeline_results):
    """Inference-only와 Pipeline 비교 막대 그래프를 생성합니다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ matplotlib가 설치되지 않아 그래프를 생성할 수 없습니다.")
        return None

    # 매칭되는 설정만 추출
    names = []
    infer_times = []
    pipeline_times = []

    for ir in infer_results:
        pr = next((p for p in pipeline_results
                   if p["_display_name"] == ir["_display_name"]), None)
        if pr:
            names.append(ir["_display_name"])
            infer_times.append(ir["avg_ms"])
            pipeline_times.append(pr["avg_ms"])

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, infer_times, width, label="Inference Only",
                   color="#4A90D9", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, pipeline_times, width, label="Full Pipeline",
                   color="#FF6B6B", edgecolor="white", linewidth=1.5)

    ax.set_ylabel("Avg Latency (ms)", fontsize=12)
    ax.set_title("Inference vs Full Pipeline Comparison", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, infer_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, pipeline_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(BENCHMARK_DIR, "comparison_pipeline.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 그래프 저장: {chart_path}")
    return chart_path


def save_combined_csv(infer_results, pipeline_results):
    """Inference + Pipeline 통합 CSV를 저장합니다."""
    csv_path = os.path.join(BENCHMARK_DIR, "comparison_full.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["설정", "벤치마크유형", "평균(ms)", "중앙값(ms)",
                         "최소(ms)", "최대(ms)", "표준편차(ms)", "반복횟수",
                         "탐지수", "측정시각"])
        for r in infer_results:
            writer.writerow([
                r["_display_name"], "inference_only",
                r["avg_ms"], r["median_ms"], r["min_ms"], r["max_ms"],
                r["std_ms"], r["runs"], "N/A", r.get("timestamp", ""),
            ])
        for r in pipeline_results:
            writer.writerow([
                r["_display_name"], "full_pipeline",
                r["avg_ms"], r["median_ms"], r["min_ms"], r["max_ms"],
                r["std_ms"], r["runs"],
                r.get("avg_detections", "N/A"), r.get("timestamp", ""),
            ])

    print(f"  📄 통합 CSV 저장: {csv_path}")
    return csv_path


def save_live_chart(results):
    """라이브 파이프라인 비교 막대 그래프를 생성합니다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib가 설치되지 않아 그래프를 생성할 수 없습니다.")
        return None

    names = [r["_display_name"] for r in results]
    avg_times = [r["avg_ms"] for r in results]
    fps_vals = [r.get("fps", 1000.0 / r["avg_ms"]) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    colors = ["#4A90D9", "#50C878", "#FF6B6B"][:len(results)]

    # 왼쪽: Latency
    bars1 = ax1.bar(names, avg_times, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Avg Pipeline Latency (ms)")
    ax1.set_title("Live Pipeline Latency")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, avg_times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}ms", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # 오른쪽: FPS
    bars2 = ax2.bar(names, fps_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("FPS")
    ax2.set_title("Live Pipeline FPS")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, fps_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(BENCHMARK_DIR, "comparison_live.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  그래프 저장: {chart_path}")
    return chart_path


def save_live_csv(results):
    """라이브 파이프라인 비교 결과를 CSV로 저장합니다."""
    csv_path = os.path.join(BENCHMARK_DIR, "comparison_live.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["설정", "평균(ms)", "중앙값(ms)", "최소(ms)", "최대(ms)",
                         "표준편차(ms)", "FPS", "프레임수", "측정시각"])
        for r in results:
            fps = r.get("fps", 1000.0 / r["avg_ms"] if r["avg_ms"] > 0 else 0)
            writer.writerow([
                r["_display_name"], r["avg_ms"], r["median_ms"],
                r["min_ms"], r["max_ms"], r["std_ms"],
                round(fps, 1), r.get("measured_frames", r.get("runs", "")),
                r.get("timestamp", ""),
            ])

    print(f"  CSV 저장: {csv_path}")
    return csv_path


def main():
    print("=" * 70)
    print("  🏁 Inference Benchmark 비교 분석")
    print("=" * 70)

    if not os.path.isdir(BENCHMARK_DIR):
        print(f"\n❌ 벤치마크 결과 폴더가 없습니다: {BENCHMARK_DIR}")
        print("   먼저 개별 벤치마크를 실행하세요.")
        sys.exit(1)

    results, missing = load_results()
    pipeline_results, pipeline_missing = load_pipeline_results()
    live_results, live_missing = load_live_results()

    if missing:
        print(f"\n  누락된 Inference 결과 ({len(missing)}개):")
        for fname, dname in missing:
            print(f"   - {dname}: {fname}")

    if pipeline_missing and live_missing:
        print(f"\n  누락된 Pipeline 결과:")
        for fname, dname in pipeline_missing:
            print(f"   - {dname}: {fname}")
        for fname, dname in live_missing:
            print(f"   - {dname}: {fname}")

    has_any = len(results) >= 2 or len(pipeline_results) >= 2 or len(live_results) >= 2
    if not has_any:
        print("\n  비교하려면 최소 2개의 결과가 필요합니다.")
        print("  실행 방법:")
        print("    python src/detection/binpicking_3d.py --device xpu")
        print("    python src/detection/binpicking_3d.py --device cpu")
        print("    python src/detection/binpicking_3d.py --device openvino")
        sys.exit(1)

    # ── Inference-only 결과 ──
    if len(results) >= 2:
        r0 = results[0]
        print(f"\n  테스트 이미지: {r0.get('image', 'N/A')} "
              f"({r0.get('image_size', 'N/A')})")
        print(f"  반복 횟수: {r0.get('runs', 'N/A')}회")

        print_individual_table(results)
        print_speedup_matrix(results)

    # ── Pipeline 결과 (저장 프레임 기반) ──
    if len(pipeline_results) >= 2:
        print_pipeline_table(pipeline_results,
                             "Full Pipeline 성능 (저장 프레임 기반)")
        print_speedup_matrix(pipeline_results)

    # ── 라이브 파이프라인 결과 ──
    if len(live_results) >= 2:
        print_pipeline_table(live_results,
                             "Live Pipeline 성능 (실시간 카메라)")
        print_speedup_matrix(live_results)

    # ── Inference vs Pipeline 비교 ──
    if results and pipeline_results:
        print_combined_comparison(results, pipeline_results)

    # ── 저장 ──
    all_for_csv = results + pipeline_results + live_results
    if len(results) >= 2:
        save_csv(results)
        save_chart(results)

    if results and pipeline_results:
        save_combined_csv(results, pipeline_results)
        save_combined_chart(results, pipeline_results)

    if len(live_results) >= 2:
        save_live_csv(live_results)
        save_live_chart(live_results)

    print("\n  비교 분석 완료!")


if __name__ == "__main__":
    main()

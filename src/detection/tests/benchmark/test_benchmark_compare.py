"""
벤치마크 결과 비교 분석
========================

개별 벤치마크(CPU, OpenVINO, GPU)의 JSON 결과를 모아서
비교 테이블, Speedup Matrix, 막대 그래프를 생성합니다.

사전 준비:
    아래 스크립트를 각각 **따로** 실행하세요:
    - test_benchmark_cpu.py
    - test_benchmark_openvino.py
    - test_benchmark_gpu.py

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
    ("result_gpu.json", "GPU (CUDA)"),
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
        elif "GPU" in name:
            short_names.append("GPU")
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
    ax.set_ylabel("평균 추론 시간 (ms)", fontsize=12)
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


def main():
    print("=" * 70)
    print("  🏁 Inference Benchmark 비교 분석")
    print("=" * 70)

    if not os.path.isdir(BENCHMARK_DIR):
        print(f"\n❌ 벤치마크 결과 폴더가 없습니다: {BENCHMARK_DIR}")
        print("   먼저 개별 벤치마크를 실행하세요.")
        sys.exit(1)

    results, missing = load_results()

    if missing:
        print(f"\n⚠️  누락된 결과 파일 ({len(missing)}개):")
        for fname, dname in missing:
            print(f"   - {dname}: {fname}")
        print("   해당 벤치마크를 먼저 실행하세요.\n")

    if len(results) < 2:
        print("❌ 비교하려면 최소 2개의 결과가 필요합니다.")
        sys.exit(1)

    # 공통 정보
    r0 = results[0]
    print(f"\n  📷 테스트 이미지: {r0.get('image', 'N/A')} ({r0.get('image_size', 'N/A')})")
    print(f"  🔁 반복 횟수: {r0.get('runs', 'N/A')}회")

    # 1) 개별 결과 테이블
    print_individual_table(results)

    # 2) Speedup Matrix
    print_speedup_matrix(results)

    # 3) CSV 저장
    save_csv(results)

    # 4) 그래프 저장
    save_chart(results)

    print("\n✅ 비교 분석 완료!")


if __name__ == "__main__":
    main()

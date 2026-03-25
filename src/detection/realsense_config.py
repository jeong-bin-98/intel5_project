"""
RealSense D435 최적 설정
========================

Depth 정밀도 최대화를 위한 센서 설정, 필터 체인, 유틸리티.

금속 릴레이 소켓 빈픽킹에 특화된 설정:
    - Threshold filter: 작업 범위 외 depth를 하드웨어 레벨에서 제거
    - Disparity 도메인 필터링: spatial/temporal이 더 정확하게 동작
    - IR exposure 튜닝: 금속 난반사에 의한 depth 누락 방지
    - Multi-frame median: 정적 캡처 시 노이즈 대폭 감소
    - 워밍업: auto-exposure 안정화 + temporal 필터 초기화
"""

import pyrealsense2 as rs
import numpy as np

from src.detection.constants import DEPTH_MIN_MM, DEPTH_MAX_MM


# =============================================================================
# 스트림 설정
# =============================================================================

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
# D435 depth 엔진은 848x480에 최적화 (Intel 권장)
# 1280x720은 근거리에서 오히려 정밀도 하락
DEPTH_WIDTH = 848
DEPTH_HEIGHT = 480
FPS = 30

# =============================================================================
# 센서 튜닝
# =============================================================================

# Visual Preset: 3=High Accuracy, 4=High Density
# High Density: 빈 픽셀 최소화 (금속 표면에서 depth 커버리지 우선)
VISUAL_PRESET = 3

# IR 레이저 출력 최대화 (금속 난반사 환경에서 신호 강화)
LASER_POWER_MAX = True

# Exposure 제어
# 금속 표면은 IR 패턴을 정반사시켜 센서가 과포화 → depth 누락
# auto exposure를 끄고 수동 설정하면 과포화를 방지할 수 있음
# ※ 환경마다 최적값이 다르므로 처음에는 auto로 시작 후 조절 권장
EXPOSURE_AUTO = True           # True=자동(기본), False=수동
EXPOSURE_VALUE = 8500          # μs (수동 시 사용, 기본 ~33000보다 낮게)

# 워밍업 프레임 수 (auto exposure + temporal 필터 안정화)
WARMUP_FRAMES = 30

# Multi-frame median (정적 캡처용)
MEDIAN_FRAMES = 7


# =============================================================================
# 파이프라인 생성
# =============================================================================

def _reset_camera():
    """이전 프로세스가 비정상 종료된 경우 카메라를 하드웨어 리셋합니다.

    kill -9 등으로 pipeline.stop() 없이 종료되면 카메라 펌웨어가
    아직 스트리밍 중이라고 인식합니다. hardware_reset()은 USB 재연결과
    동일한 효과로 이 상태를 해제합니다.
    """
    import time
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("RealSense 카메라를 찾을 수 없습니다.")
    for dev in devices:
        dev.hardware_reset()
    print("  RealSense 하드웨어 리셋 완료, 재연결 대기...")
    time.sleep(3)


def create_pipeline(color_res=None, depth_res=None, fps=FPS):
    """RealSense 파이프라인을 생성하고 최적 설정을 적용합니다.

    센서 프리셋, 레이저 출력, exposure, 필터 체인을 모두 설정합니다.

    Args:
        color_res: (width, height) 컬러 해상도 (기본: 1280x720)
        depth_res: (width, height) depth 해상도 (기본: 1280x720)
        fps: 프레임 레이트

    Returns:
        pipeline: rs.pipeline
        align: rs.align (color 기준 정렬)
        filters: dict (필터 체인)
        intrinsics: 카메라 내부 파라미터
    """
    if color_res is None:
        color_res = (COLOR_WIDTH, COLOR_HEIGHT)
    if depth_res is None:
        depth_res = (DEPTH_WIDTH, DEPTH_HEIGHT)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_res[0], color_res[1],
                         rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_res[0], depth_res[1],
                         rs.format.z16, fps)

    # 파이프라인 시작 시도, 실패하면 카메라 리셋 후 재시도
    try:
        profile = pipeline.start(config)
    except RuntimeError:
        print("  [경고] 카메라가 잠겨 있습니다. 하드웨어 리셋 시도...")
        _reset_camera()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, color_res[0], color_res[1],
                             rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, depth_res[0], depth_res[1],
                             rs.format.z16, fps)
        profile = pipeline.start(config)

    # --- 센서 최적화 ---
    _configure_sensor(profile)

    align = rs.align(rs.stream.color)

    # --- Post-processing 필터 체인 ---
    filters = create_filters()

    # 워밍업: auto exposure 안정화 + temporal 필터 버퍼 채움
    print(f"  RealSense 워밍업 ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        if df:
            apply_filters(df, filters)

    # Intrinsics 취득
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    print(f"  Camera: {intrinsics.width}x{intrinsics.height} "
          f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}")

    return pipeline, align, filters, intrinsics


def _configure_sensor(profile):
    """Depth 센서의 하드웨어 설정을 최적화합니다."""
    depth_sensor = profile.get_device().first_depth_sensor()

    # 1) Visual Preset
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, VISUAL_PRESET)
        preset_names = {3: "High Accuracy", 4: "High Density"}
        print(f"  Visual Preset: {preset_names.get(VISUAL_PRESET, VISUAL_PRESET)}")

    # 2) IR 레이저 출력
    if LASER_POWER_MAX and depth_sensor.supports(rs.option.laser_power):
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, laser_range.max)
        print(f"  Laser Power: {laser_range.max:.0f} (max)")

    # 3) Exposure 제어
    if not EXPOSURE_AUTO:
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, EXPOSURE_VALUE)
        print(f"  Exposure: manual {EXPOSURE_VALUE}μs")
    else:
        print(f"  Exposure: auto")


# =============================================================================
# Post-processing 필터 체인
# =============================================================================

def create_filters():
    """최적화된 depth 후처리 필터 체인을 생성합니다.

    Intel 권장 순서:
        Threshold → Depth→Disparity → Spatial → Temporal → Disparity→Depth → Hole

    Threshold: 작업 범위(100~1000mm) 외 depth를 즉시 제거
    Disparity 도메인: spatial/temporal 필터가 거리에 비례하는 noise에 적응적으로 동작
    """
    # 1) Threshold: 작업 범위 외 depth 즉시 제거
    #    배경/테이블/벽 등의 depth가 좌표 계산에 간섭하는 것을 차단
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, DEPTH_MIN_MM / 1000.0)
    threshold.set_option(rs.option.max_distance, DEPTH_MAX_MM / 1000.0)

    # 2) Depth → Disparity 변환
    #    Spatial/Temporal이 disparity 도메인에서 더 정확 (근거리 노이즈 적응적 처리)
    depth_to_disparity = rs.disparity_transform(True)

    # 3) Spatial: 공간축 엣지 보존 스무딩
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 3)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    # 4) Temporal: 시간축 스무딩 (정적/준정적 장면에서 매우 효과적)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    # 5) Disparity → Depth 역변환
    disparity_to_depth = rs.disparity_transform(False)

    # 6) Hole filling: 빈 픽셀을 인접값으로 보간
    hole_filling = rs.hole_filling_filter()

    return {
        'threshold': threshold,
        'depth_to_disparity': depth_to_disparity,
        'spatial': spatial,
        'temporal': temporal,
        'disparity_to_depth': disparity_to_depth,
        'hole_filling': hole_filling,
    }


def apply_filters(depth_frame, filters):
    """필터 체인을 Intel 권장 순서대로 적용합니다.

    Args:
        depth_frame: rs.frame (depth)
        filters: create_filters()에서 반환된 dict

    Returns:
        filtered depth_frame
    """
    frame = filters['threshold'].process(depth_frame)
    frame = filters['depth_to_disparity'].process(frame)
    frame = filters['spatial'].process(frame)
    frame = filters['temporal'].process(frame)
    frame = filters['disparity_to_depth'].process(frame)
    frame = filters['hole_filling'].process(frame)
    return frame


# =============================================================================
# Multi-frame Median 캡처
# =============================================================================

def capture_median_depth(pipeline, align, filters, n_frames=MEDIAN_FRAMES):
    """N프레임의 중앙값 depth 이미지를 생성합니다.

    정적 장면(빈픽킹 캡처 시)에서 단일 프레임 대비 depth 노이즈를
    크게 줄입니다. Temporal filter 위에 추가 적용하면 더 효과적.

    단일 프레임 depth는 랜덤 노이즈가 있지만, N프레임 중앙값은
    outlier를 제거하여 안정적인 값을 얻습니다.

    Args:
        pipeline: rs.pipeline
        align: rs.align
        filters: 필터 체인
        n_frames: 수집할 프레임 수 (기본 7)

    Returns:
        color_image: 마지막 컬러 프레임
        depth_image: 중앙값 depth (uint16, mm)
        intrinsics: 카메라 내부 파라미터
    """
    depth_stack = []
    color_image = None
    intrinsics = None

    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        if intrinsics is None:
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # 필터 적용
        depth_frame = apply_filters(depth_frame, filters)

        depth_stack.append(np.asanyarray(depth_frame.get_data()).copy())
        color_image = np.asanyarray(color_frame.get_data())

    if not depth_stack:
        return None, None, None

    # 픽셀별 중앙값 → outlier 제거
    depth_median = np.median(np.stack(depth_stack), axis=0).astype(np.uint16)

    return color_image, depth_median, intrinsics

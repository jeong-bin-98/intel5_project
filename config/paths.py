"""
프로젝트 경로 설정
==================
모든 스크립트에서 공통으로 사용하는 경로를 정의합니다.
프로젝트 루트 기준 절대 경로로 관리하여, 어디서 실행하든 동일하게 동작합니다.
"""

import os

# 프로젝트 루트 디렉토리 (config/ 의 상위)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 데이터 디렉토리 =====
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CAPTURES_DIR = os.path.join(DATA_DIR, "raw_captures")
CLEANED_IMAGES_DIR = os.path.join(DATA_DIR, "cleaned_images")
NEW_IMAGES_DIR = os.path.join(DATA_DIR, "new_images")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATASET_IMAGES_TRAIN = os.path.join(DATASET_DIR, "images", "train")
DATASET_IMAGES_VAL = os.path.join(DATASET_DIR, "images", "val")
DATASET_LABELS_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
DATASET_LABELS_VAL = os.path.join(DATASET_DIR, "labels", "val")

# ===== 모델 가중치 =====
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAM_VIT_H = os.path.join(MODELS_DIR, "sam_vit_h_4b8939.pth")
SAM_VIT_B = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
YOLOV8N_PT = os.path.join(MODELS_DIR, "yolov8n.pt")

# ===== YOLO 학습 결과 =====
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
YOLO_BEST_PT = os.path.join(RUNS_DIR, "detect", "socket_detector_obb4", "weights", "best.pt")

# ===== OBB 데이터셋 (이미지는 심볼릭 링크, 라벨은 별도 저장) =====
# YOLO는 이미지 경로의 /images/ → /labels/ 자동 치환으로 라벨을 찾으므로
# dataset_obb/images/ + dataset_obb/labels/ 구조를 사용
DATASET_OBB_DIR = os.path.join(DATA_DIR, "dataset_obb")
DATASET_LABELS_OBB_TRAIN = os.path.join(DATASET_OBB_DIR, "labels", "train")
DATASET_LABELS_OBB_VAL = os.path.join(DATASET_OBB_DIR, "labels", "val")
DATASET_IMAGES_OBB_VAL = os.path.join(DATASET_OBB_DIR, "images", "val")

# ===== 벤치마크 결과 =====
BENCHMARK_DIR = os.path.join(RUNS_DIR, "benchmark")

# ===== 설정 파일 =====
DATASET_YAML = os.path.join(PROJECT_ROOT, "dataset.yaml")
DATASET_OBB_YAML = os.path.join(PROJECT_ROOT, "dataset_obb.yaml")

# ===== 로봇 SDK =====
ROBOT_SDK_DIR = os.path.join(PROJECT_ROOT, "robot")

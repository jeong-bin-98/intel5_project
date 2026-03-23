"""
Step 2 (OBB): SAM + 수동 회전 바운딩박스(OBB) 라벨링 도구

사용법:
    python step2_obb_labeler.py           # vit_b (빠름)
    python step2_obb_labeler.py --heavy   # vit_h (정확)

조작법:
    [SAM 모드 - 기본]
    - 마우스 왼쪽 클릭  : 소켓 위를 클릭 → SAM이 윤곽 자동 탐지 + 회전박스 계산

    [수동 모드 - 'm' 키로 전환]
    - 마우스 왼쪽 클릭  : 코너 점 4개를 순서대로 클릭 → OBB 완성

    [공통]
    - '1' 키  : 현재 박스를 "8pin"  (클래스 0)으로 저장
    - '2' 키  : 현재 박스를 "12pin" (클래스 1)으로 저장
    - 'u' 키  : 마지막 저장 라벨 취소 (undo)
    - 'r' 키  : 현재 작업 리셋
    - 'm' 키  : SAM 모드 ↔ 수동 모드 전환
    - 't' 키  : 현재 이미지 라벨 저장 후 다음 이미지
    - 's' 키  : 현재 이미지 스킵 (라벨 없이 넘김)
    - 'b' 키  : 이전 이미지로 돌아가기 (기존 라벨 자동 복원)
    - 'q' 키  : 저장 후 종료

출력:
    YOLO OBB 형식 라벨 파일 (.txt)
    각 줄: class x1 y1 x2 y2 x3 y3 x4 y4  (정규화 좌표, 4 코너)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.paths import (
    SAM_VIT_H, SAM_VIT_B,
    RAW_CAPTURES_DIR,
    DATASET_IMAGES_TRAIN, DATASET_IMAGES_VAL,
    DATASET_OBB_DIR,
    DATASET_LABELS_OBB_TRAIN, DATASET_LABELS_OBB_VAL,
)

import glob
import shutil
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor


# ===== 설정 =====
if "--heavy" in sys.argv:
    SAM_CHECKPOINT = SAM_VIT_H
    SAM_MODEL_TYPE = "vit_h"
else:
    SAM_CHECKPOINT = SAM_VIT_B
    SAM_MODEL_TYPE = "vit_b"

# 화면에 표시할 최대 크기 (이 안에 이미지가 fit되도록 스케일)
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 800

CLASS_NAMES  = {0: "8pin", 1: "12pin"}
CLASS_COLORS = {0: (255, 100, 0), 1: (0, 100, 255)}  # BGR

# ===== 전역 상태 =====
current_mask   = None
current_obb    = None       # np.array shape (8,), 정규화 좌표
current_image  = None
predictor      = None
manual_mode    = False
manual_points  = []         # 수동 모드 클릭 점 (원본 픽셀 좌표)
mouse_pos_orig = (0, 0)     # 마우스 위치 (원본 좌표계)
display_scale  = 1.0        # 디스플레이 축소 비율


def orig_coord(dx, dy):
    """디스플레이 좌표 → 원본 이미지 좌표"""
    return int(dx / display_scale), int(dy / display_scale)


def mouse_callback(event, x, y, flags, param):
    global current_mask, current_obb, manual_points, mouse_pos_orig

    ox, oy = orig_coord(x, y)
    mouse_pos_orig = (ox, oy)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if manual_mode:
        manual_points.append((ox, oy))
        print(f"  pt {len(manual_points)}/4  ({ox}, {oy})")
        if len(manual_points) == 4:
            h, w = current_image.shape[:2]
            current_obb = points_to_obb(manual_points, h, w)
            manual_points.clear()
            if current_obb is not None:
                print("  4pts done -> OBB ready  '1'=8pin  '2'=12pin  'r'=reset")
            else:
                print("  OBB failed, try again")
    else:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[ox, oy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = np.argmax(scores)
        current_mask = masks[best]

        h, w = current_image.shape[:2]
        current_obb = mask_to_obb(current_mask, h, w)

        if current_obb is not None:
            print(f"  click ({ox},{oy}) score={scores[best]:.3f}  '1'=8pin  '2'=12pin  'r'=reset")
        else:
            print(f"  click ({ox},{oy}) mask too small, try again")


# ───────────────────────────────────────────────
#  변환 함수
# ───────────────────────────────────────────────

def mask_to_obb(mask, img_h, img_w):
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    pts   = np.column_stack([xs, ys]).astype(np.float32)
    rect  = cv2.minAreaRect(pts)
    box   = cv2.boxPoints(rect)
    box[:, 0] /= img_w
    box[:, 1] /= img_h
    return np.clip(box, 0.0, 1.0).flatten()


def points_to_obb(pts_px, img_h, img_w):
    if len(pts_px) != 4:
        return None
    box = np.array(pts_px, dtype=np.float32)
    box[:, 0] /= img_w
    box[:, 1] /= img_h
    return np.clip(box, 0.0, 1.0).flatten()


# ───────────────────────────────────────────────
#  그리기 함수  (원본 좌표 → 디스플레이 좌표 변환 포함)
# ───────────────────────────────────────────────

def to_disp(x_norm, y_norm, orig_w, orig_h):
    """정규화 좌표 → 디스플레이 픽셀 좌표"""
    return (int(x_norm * orig_w * display_scale),
            int(y_norm * orig_h * display_scale))


def draw_obb(img, obb_norm, orig_h, orig_w, color, label_text):
    coords = obb_norm.reshape(4, 2)
    pts = np.array([to_disp(c[0], c[1], orig_w, orig_h) for c in coords], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    cv2.putText(img, label_text, tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_mask_overlay(img, mask):
    """원본 마스크를 디스플레이 크기로 리사이즈해서 오버레이"""
    dh, dw = img.shape[:2]
    mask_small = cv2.resize(mask.astype(np.uint8), (dw, dh),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = img.copy()
    overlay[mask_small] = [0, 220, 0]
    return cv2.addWeighted(img, 0.65, overlay, 0.35, 0)


def draw_manual_preview(img, pts_orig, orig_h, orig_w):
    colors_pt = [(0, 255, 255), (0, 200, 255), (0, 150, 255), (0, 100, 255)]
    disp_pts  = [(int(p[0] * display_scale), int(p[1] * display_scale))
                 for p in pts_orig]
    mouse_d   = (int(mouse_pos_orig[0] * display_scale),
                 int(mouse_pos_orig[1] * display_scale))

    for i, dp in enumerate(disp_pts):
        cv2.circle(img, dp, 7, colors_pt[i], -1)
        cv2.putText(img, str(i + 1), (dp[0] + 8, dp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_pt[i], 2)
        if i > 0:
            cv2.line(img, disp_pts[i - 1], dp, (0, 220, 220), 1)

    if disp_pts:
        cv2.line(img, disp_pts[-1], mouse_d, (0, 220, 220), 1, cv2.LINE_AA)
        if len(disp_pts) == 3:
            cv2.line(img, disp_pts[0], mouse_d, (0, 220, 220), 1, cv2.LINE_AA)


def draw_info_bar(img, img_idx, total, label_count, reset_flash):
    h, w = img.shape[:2]
    mode_str = "[MANUAL]" if manual_mode else "[SAM]   "
    line1 = (f"{mode_str}  [{img_idx+1}/{total}]  labels:{label_count}"
             f"  | 1=8pin  2=12pin  u=undo  r=reset  m=mode")
    line2 = "t=save+next  s=skip  b=back  q=quit"
    cv2.rectangle(img, (0, 0), (w, 56), (30, 30, 30), -1)
    color = (0, 220, 220) if manual_mode else (220, 220, 220)
    cv2.putText(img, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
    cv2.putText(img, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

    if reset_flash > 0:
        cv2.putText(img, "RESET", (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


# ───────────────────────────────────────────────
#  초기화
# ───────────────────────────────────────────────

def compute_scale(img_h, img_w):
    global display_scale
    display_scale = min(DISPLAY_MAX_W / img_w, DISPLAY_MAX_H / img_h, 1.0)


def get_display_image(img):
    if display_scale < 1.0:
        dw = int(img.shape[1] * display_scale)
        dh = int(img.shape[0] * display_scale)
        return cv2.resize(img, (dw, dh))
    return img.copy()


def reset_current():
    global current_mask, current_obb, manual_points
    current_mask  = None
    current_obb   = None
    manual_points = []


def import_raw_captures():
    """
    data/raw_captures/color/ 의 새 이미지를 data/dataset/images/train/ 으로 복사합니다.
    이미 복사된 파일(동일 파일명)은 건너뜁니다.
    """
    src_dir = os.path.join(RAW_CAPTURES_DIR, "color")
    if not os.path.exists(src_dir):
        return

    os.makedirs(DATASET_IMAGES_TRAIN, exist_ok=True)
    src_files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    copied = 0
    for src in src_files:
        dst = os.path.join(DATASET_IMAGES_TRAIN, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    if copied:
        print(f"raw_captures → dataset/images/train: {copied}장 복사됨")
    else:
        print(f"raw_captures → dataset/images/train: 새 이미지 없음 (총 {len(src_files)}장 이미 존재)")


def sync_obb_images():
    """
    data/dataset/images/ → data/dataset_obb/images/ 로 하드링크 동기화.
    YOLO가 symlink를 resolve해서 /images/→/labels/ 매핑이 깨지므로 하드링크를 사용.
    새 이미지만 추가하고, 이미 있는 파일은 건너뜀.
    """
    for name, src_dir in [("train", DATASET_IMAGES_TRAIN), ("val", DATASET_IMAGES_VAL)]:
        dst_dir = os.path.join(DATASET_OBB_DIR, "images", name)
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(src_dir):
            continue
        linked = 0
        for fname in os.listdir(src_dir):
            if not fname.endswith(".png"):
                continue
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                os.link(os.path.join(src_dir, fname), dst)
                linked += 1
        if linked:
            print(f"  dataset_obb/images/{name}: {linked}장 하드링크 추가")


# ───────────────────────────────────────────────
#  메인
# ───────────────────────────────────────────────

def load_existing_labels(lbl_path):
    """저장된 라벨 파일을 읽어 current_labels 형식으로 반환"""
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    cls_id = int(parts[0])
                    obb = np.array([float(v) for v in parts[1:]])
                    labels.append((cls_id, obb))
    return labels


def main():
    global current_image, predictor, manual_mode

    os.makedirs(DATASET_IMAGES_TRAIN, exist_ok=True)
    os.makedirs(DATASET_LABELS_OBB_TRAIN, exist_ok=True)
    os.makedirs(DATASET_LABELS_OBB_VAL, exist_ok=True)
    import_raw_captures()
    sync_obb_images()

    print(f"\nLoading SAM ({SAM_MODEL_TYPE})...")
    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"SAM ready (device: {device})\n")

    all_images = sorted(glob.glob(f"{DATASET_IMAGES_TRAIN}/*.png"))
    pending, done_count = [], 0
    for img_path in all_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        if os.path.exists(os.path.join(DATASET_LABELS_OBB_TRAIN, f"{base}.txt")):
            done_count += 1
        else:
            pending.append(img_path)

    print(f"Total:{len(all_images)}  Done:{done_count}  Remaining:{len(pending)}\n")
    if not pending:
        print("All images labeled!")
        return

    cv2.namedWindow("OBB Labeler", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("OBB Labeler", mouse_callback)

    img_idx = 0
    while img_idx < len(pending):
        img_path = pending[img_idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{img_idx+1}/{len(pending)}] {basename}")

        current_image = cv2.imread(img_path)
        img_h, img_w  = current_image.shape[:2]
        compute_scale(img_h, img_w)

        img_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        # 기존 라벨이 있으면 복원 (뒤로가기로 돌아온 경우)
        lbl_path = os.path.join(DATASET_LABELS_OBB_TRAIN, f"{basename}.txt")
        current_labels = load_existing_labels(lbl_path)
        if current_labels:
            print(f"  (기존 라벨 {len(current_labels)}개 복원)")
        reset_current()
        reset_flash = 0   # 'r' 눌렀을 때 화면에 RESET 텍스트 표시용 카운터

        action = 'next'   # 'next' | 'back' | 'quit'

        while True:
            display = get_display_image(current_image)

            if current_mask is not None:
                display = draw_mask_overlay(display, current_mask)

            if manual_mode and manual_points:
                draw_manual_preview(display, manual_points, img_h, img_w)

            if current_obb is not None:
                draw_obb(display, current_obb, img_h, img_w, (0, 220, 0), "? 1=8pin 2=12pin")

            for cls_id, obb in current_labels:
                draw_obb(display, obb, img_h, img_w, CLASS_COLORS[cls_id], CLASS_NAMES[cls_id])

            draw_info_bar(display, img_idx, len(pending), len(current_labels), reset_flash)
            if reset_flash > 0:
                reset_flash -= 1

            cv2.imshow("OBB Labeler", display)
            key = cv2.waitKey(30) & 0xFF

            # ── reset (가장 먼저 단독 if로 체크) ──
            # r = 현재 작업 중인 박스 + 이미 저장한 라벨 전체 초기화
            if key == ord('r'):
                reset_current()
                current_labels.clear()
                reset_flash = 10
                print("    reset all")
                continue

            # ── 클래스 저장 ──
            if key in (ord('1'), ord('2')) and current_obb is not None:
                cls_id = 0 if key == ord('1') else 1
                current_labels.append((cls_id, current_obb.copy()))
                print(f"    -> {CLASS_NAMES[cls_id]} saved (total {len(current_labels)})")
                reset_current()

            elif key == ord('u'):
                if current_labels:
                    removed = current_labels.pop()
                    print(f"    undo: {CLASS_NAMES[removed[0]]} removed ({len(current_labels)} left)")
                reset_current()

            elif key == ord('m'):
                manual_mode = not manual_mode
                reset_current()
                print(f"    mode -> {'MANUAL (click 4 corners)' if manual_mode else 'SAM (click center)'}")

            elif key == ord('t'):
                with open(lbl_path, 'w') as f:
                    for cls_id, obb in current_labels:
                        coords = " ".join(f"{v:.6f}" for v in obb)
                        f.write(f"{cls_id} {coords}\n")
                print(f"  saved {len(current_labels)} labels -> {lbl_path}")
                action = 'next'
                break

            elif key == ord('s'):
                print("  skipped")
                action = 'next'
                break

            elif key == ord('b'):
                if img_idx > 0:
                    action = 'back'
                    reset_current()
                    print("    <- back")
                    break
                else:
                    print("    already at first image")

            elif key == ord('q'):
                action = 'quit'
                break

        if action == 'back':
            img_idx -= 1
        elif action == 'quit':
            cv2.destroyAllWindows()
            saved = sum(
                1 for p in pending
                if os.path.exists(os.path.join(
                    DATASET_LABELS_OBB_TRAIN,
                    os.path.splitext(os.path.basename(p))[0] + ".txt"
                ))
            )
            print(f"\nQuit. This session: {saved} images saved.")
            return
        else:
            img_idx += 1

    cv2.destroyAllWindows()
    print(f"\nAll done!")
    print(f"  Labels: {DATASET_LABELS_OBB_TRAIN}")
    print(f"  Next:   python step3_train_yolo_obb.py")


if __name__ == "__main__":
    main()
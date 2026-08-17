"""使用 MediaPipe 从摄像头实时检测并绘制手部关键点。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"
WRIST_LANDMARK_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时识别摄像头画面中的手部")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认为 0")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"手部识别模型路径，默认为 {DEFAULT_MODEL}",
    )
    parser.add_argument("--max-hands", type=int, default=2, help="最多检测的手数")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="最低检测置信度",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="最低跟踪置信度",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        print(
            f"找不到模型文件：{args.model}\n"
            "请重新下载 hand_landmarker.task，或通过 --model 指定模型路径。",
            file=sys.stderr,
        )
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(
            f"无法打开摄像头 {args.camera}，请检查摄像头权限或尝试 --camera 1。",
            file=sys.stderr,
        )
        return 1

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    landmark_style = vision.drawing_utils.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=3
    )
    connection_style = vision.drawing_utils.DrawingSpec(
        color=(255, 180, 0), thickness=2
    )

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            start_time = time.monotonic()
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("无法读取摄像头画面。", file=sys.stderr)
                    break

                # 镜像显示更符合用户面对摄像头时的操作习惯。
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    for index, landmarks in enumerate(result.hand_landmarks):
                        vision.drawing_utils.draw_landmarks(
                            frame,
                            landmarks,
                            vision.HandLandmarksConnections.HAND_CONNECTIONS,
                            landmark_style,
                            connection_style,
                        )

                        if index < len(result.handedness) and result.handedness[index]:
                            classification = result.handedness[index][0]
                            # MediaPipe 的 21 个手部关键点中，索引 0 固定为手腕。
                            wrist = landmarks[WRIST_LANDMARK_INDEX]
                            height, width = frame.shape[:2]
                            position = (
                                max(0, int(wrist.x * width) - 30),
                                max(30, int(wrist.y * height) - 20),
                            )
                            label = (
                                f"{classification.category_name} "
                                f"{classification.score:.2f}"
                            )
                            cv2.putText(
                                frame,
                                label,
                                position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                cv2.putText(
                    frame,
                    "Press Q or ESC to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("MediaPipe Hand Tracking", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    if args.max_hands < 1:
        print("--max-hands 必须大于 0。", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

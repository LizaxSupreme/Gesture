"""追踪拇指与食指捏合期间的接触点轨迹。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "trajectories"
DELETE_COOLDOWN_SECONDS = 1.5
GRAB_RELEASE_COOLDOWN_SECONDS = 1.5
WAVE_WINDOW_SECONDS = 0.9
WAVE_COOLDOWN_SECONDS = 1.2
WAVE_MISSING_TOLERANCE_SECONDS = 0.3
CLAP_MISSING_TOLERANCE_SECONDS = 0.3

# MediaPipe Hands 的固定关键点索引。
WRIST = 0
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20


@dataclass
class Trajectory:
    """画布中可移动、可缩放的一条轨迹。"""

    points: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="记录拇指和食指捏合时的运动轨迹")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument(
        "--mode",
        type=str.upper,
        choices=("WHITEBOARD", "AR"),
        default="WHITEBOARD",
        help="显示模式：WHITEBOARD（默认白板）或 AR（视频叠加）",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="模型路径")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="CSV 输出目录"
    )
    parser.add_argument(
        "--pinch-threshold",
        type=float,
        default=0.04,
        help="开始捏合阈值（指尖距离/画面短边，默认 0.04）",
    )
    parser.add_argument(
        "--release-threshold",
        type=float,
        default=0.06,
        help="结束捏合阈值（指尖距离/画面短边，默认 0.06）",
    )
    parser.add_argument(
        "--movement-gain",
        type=float,
        default=2.0,
        help="手部移动映射到画布的放大倍数（默认 2.0）",
    )
    return parser.parse_args()


def distance(a: object, b: object) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_center(landmarks: list[object]) -> tuple[float, float]:
    indices = (WRIST, INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP)
    return (
        sum(landmarks[index].x for index in indices) / len(indices),
        sum(landmarks[index].y for index in indices) / len(indices),
    )


def amplify_point(
    point: tuple[float, float], gain: float
) -> tuple[float, float]:
    """以画面中心为原点放大位移，并限制在画布范围内。"""
    return (
        min(1.0, max(0.0, 0.5 + (point[0] - 0.5) * gain)),
        min(1.0, max(0.0, 0.5 + (point[1] - 0.5) * gain)),
    )


def is_five_finger_grab(landmarks: list[object], palm_width: float) -> bool:
    """五指收拢（握拳）时视为抓取。"""
    wrist = landmarks[WRIST]
    folded_fingers = all(
        distance(landmarks[tip], wrist) < distance(landmarks[pip], wrist)
        for tip, pip in (
            (INDEX_FINGER_TIP, INDEX_FINGER_PIP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP),
            (RING_FINGER_TIP, RING_FINGER_PIP),
            (PINKY_TIP, PINKY_PIP),
        )
    )
    center_x, center_y = palm_center(landmarks)
    thumb = landmarks[THUMB_TIP]
    thumb_folded = math.hypot(thumb.x - center_x, thumb.y - center_y) < palm_width
    return folded_fingers and thumb_folded


def nearest_trajectory(
    trajectories: list[Trajectory],
    center: tuple[float, float],
    width: int,
    height: int,
    maximum_distance: float = 80.0,
) -> int | None:
    """返回手掌附近最近的轨迹编号。"""
    center_x, center_y = center
    best_index = None
    best_distance = maximum_distance
    for index, item in enumerate(trajectories):
        for point_x, point_y in item.points:
            point_distance = math.hypot(
                (point_x - center_x) * width, (point_y - center_y) * height
            )
            if point_distance < best_distance:
                best_distance = point_distance
                best_index = index
    return best_index


def in_delete_corner(center: tuple[float, float], width: int, height: int) -> bool:
    zone = min(120, round(min(width, height) * 0.18))
    x, y = round(center[0] * width), round(center[1] * height)
    near_horizontal_edge = x <= zone or x >= width - zone
    near_vertical_edge = y <= zone or y >= height - zone
    return near_horizontal_edge and near_vertical_edge


def draw_delete_zones(frame: object) -> None:
    height, width = frame.shape[:2]
    zone = min(120, round(min(width, height) * 0.18))
    color = (70, 70, 220)
    for start, end in (
        ((0, 0), (zone, zone)),
        ((width - zone, 0), (width - 1, zone)),
        ((0, height - zone), (zone, height - 1)),
        ((width - zone, height - zone), (width - 1, height - 1)),
    ):
        cv2.rectangle(frame, start, end, color, 2)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """打开摄像头；macOS 优先使用 AVFoundation 后端。"""
    if sys.platform == "darwin":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(camera_index)


class CameraStream:
    """在后台持续读取最新摄像头帧，并在断流后自动重连。"""

    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self.capture = open_camera(camera_index)
        self.frame = None
        self.sequence = 0
        self.last_frame_at = float("-inf")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _update(self) -> None:
        while not self.stop_event.is_set():
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = open_camera(self.camera_index)
                if not self.capture.isOpened():
                    self.stop_event.wait(0.25)
                    continue

            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.last_frame_at = time.monotonic()
                continue

            self.capture.release()
            self.stop_event.wait(0.2)

    def read(self) -> tuple[object | None, int, float]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, self.sequence, self.last_frame_at

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.capture.release()


def save_trajectory(
    points: list[tuple[float, float, float]], output_dir: Path
) -> Path | None:
    if not points:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("pinch_%Y%m%d_%H%M%S_%f.csv")
    path = output_dir / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("elapsed_seconds", "x_normalized", "y_normalized"))
        writer.writerows(points)
    return path


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        print(f"找不到模型文件：{args.model}", file=sys.stderr)
        return 1

    camera_stream = CameraStream(args.camera)
    camera_stream.start()

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recording = False
    trajectory: list[tuple[float, float, float]] = []
    active_points: list[tuple[float, float]] = []
    trajectories: list[Trajectory] = []
    recording_started_at = 0.0
    grabbed_index: int | None = None
    grab_original_points: list[tuple[float, float]] = []
    grab_started_at = (0.0, 0.0)
    grab_hand_size = 1.0
    was_grabbing = False
    clap_armed = False
    previous_hand_x: tuple[float, float] | None = None
    last_two_hands_seen_at = float("-inf")
    last_two_hand_gap = float("inf")
    clap_inward_motion_seen = False
    last_clear_at = float("-inf")
    last_delete_at = float("-inf")
    last_grab_release_at = float("-inf")
    system_enabled = False
    wave_samples: list[tuple[float, float, float]] = []
    last_wave_seen_at = float("-inf")
    last_toggle_at = float("-inf")
    status_message = "SYSTEM OFF - swipe left to right"
    last_frame_sequence = -1
    last_camera_warning_at = float("-inf")

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            video_started_at = time.monotonic()
            while True:
                frame, frame_sequence, last_frame_at = camera_stream.read()
                if frame is None or frame_sequence == last_frame_sequence:
                    now = time.monotonic()
                    if now - last_frame_at > 1.0 and now - last_camera_warning_at > 3.0:
                        print(
                            f"摄像头 {args.camera} 暂时无新画面，正在后台重连……",
                            file=sys.stderr,
                        )
                        last_camera_warning_at = now
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    time.sleep(0.01)
                    continue
                last_frame_sequence = frame_sequence

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                if args.mode == "AR":
                    canvas = frame
                    ui_color = (255, 255, 255)
                else:
                    canvas = frame.copy()
                    canvas.fill(255)
                    ui_color = (40, 40, 40)
                now = time.monotonic()
                timestamp_ms = int((now - video_started_at) * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)
                detected_hand_count = len(result.hand_landmarks)

                # 单手在限定时间内横跨画面，用作识别系统的总开关。
                toggle_changed = False
                if len(result.hand_landmarks) == 1:
                    wave_x, wave_y = amplify_point(
                        palm_center(result.hand_landmarks[0]), args.movement_gain
                    )
                    last_wave_seen_at = now
                    wave_samples.append((now, wave_x, wave_y))
                    wave_samples = [
                        sample
                        for sample in wave_samples
                        if sample[0] >= now - WAVE_WINDOW_SECONDS
                    ]
                    _, start_x, start_y = wave_samples[0]
                    enough_time_since_toggle = (
                        now - last_toggle_at >= WAVE_COOLDOWN_SECONDS
                    )
                    similar_height = abs(wave_y - start_y) <= 0.30
                    if (
                        not system_enabled
                        and enough_time_since_toggle
                        and similar_height
                        and start_x <= 0.25
                        and wave_x >= 0.75
                    ):
                        system_enabled = True
                        last_toggle_at = now
                        wave_samples.clear()
                        toggle_changed = True
                        status_message = "SYSTEM ON"
                    elif (
                        system_enabled
                        and enough_time_since_toggle
                        and similar_height
                        and start_x >= 0.75
                        and wave_x <= 0.25
                    ):
                        system_enabled = False
                        last_toggle_at = now
                        wave_samples.clear()
                        toggle_changed = True
                        if recording and active_points:
                            trajectories.append(Trajectory(active_points[:]))
                        recording = False
                        active_points = []
                        trajectory = []
                        grabbed_index = None
                        grab_original_points = []
                        was_grabbing = False
                        clap_armed = False
                        status_message = "SYSTEM OFF"
                elif len(result.hand_landmarks) >= 2:
                    # 双手动作不应沿用单手挥动的历史。
                    wave_samples.clear()
                elif now - last_wave_seen_at > WAVE_MISSING_TOLERANCE_SECONDS:
                    # 快速挥动可能造成短暂漏检；仅在持续丢失后重置。
                    wave_samples.clear()

                delete_cooldown_remaining = DELETE_COOLDOWN_SECONDS - (
                    now - last_delete_at
                )
                delete_cooldown_active = delete_cooldown_remaining > 0
                grab_cooldown_remaining = GRAB_RELEASE_COOLDOWN_SECONDS - (
                    now - last_grab_release_at
                )
                grab_cooldown_active = grab_cooldown_remaining > 0
                if system_enabled and delete_cooldown_active:
                    status_message = (
                        f"Delete cooldown: {delete_cooldown_remaining:.1f}s"
                    )
                elif system_enabled and grab_cooldown_active:
                    status_message = (
                        f"Move cooldown: {grab_cooldown_remaining:.1f}s"
                    )
                elif not system_enabled and not toggle_changed:
                    status_message = "SYSTEM OFF - swipe left to right"

                hand_visible = bool(result.hand_landmarks)
                clap_cleared = False
                if system_enabled and len(result.hand_landmarks) >= 2:
                    last_two_hands_seen_at = now
                    centers = sorted(
                        (
                            amplify_point(palm_center(hand), args.movement_gain)
                            for hand in result.hand_landmarks[:2]
                        ),
                        key=lambda point: point[0],
                    )
                    left_center, right_center = centers
                    horizontal_gap = right_center[0] - left_center[0]

                    # 两只手分别出现在左右两侧时，准备识别一次击掌。
                    if (
                        horizontal_gap >= 0.45
                        and left_center[0] <= 0.40
                        and right_center[0] >= 0.60
                    ):
                        if not clap_armed:
                            clap_inward_motion_seen = False
                        clap_armed = True

                    moving_inward = False
                    if previous_hand_x is not None:
                        left_velocity = left_center[0] - previous_hand_x[0]
                        right_velocity = previous_hand_x[1] - right_center[0]
                        moving_inward = left_velocity >= 0.004 and right_velocity >= 0.004
                        if clap_armed and moving_inward:
                            clap_inward_motion_seen = True

                    same_height = abs(left_center[1] - right_center[1]) <= 0.30
                    if (
                        clap_armed
                        and horizontal_gap <= 0.18
                        and moving_inward
                        and same_height
                        and now - last_clear_at >= 1.5
                    ):
                        trajectories.clear()
                        active_points.clear()
                        trajectory.clear()
                        recording = False
                        grabbed_index = None
                        grab_original_points = []
                        was_grabbing = False
                        clap_armed = False
                        last_clear_at = now
                        clap_cleared = True
                        status_message = "CLAP - canvas cleared"

                    previous_hand_x = (left_center[0], right_center[0])
                    last_two_hand_gap = horizontal_gap
                elif (
                    system_enabled
                    and len(result.hand_landmarks) == 1
                    and clap_armed
                    and clap_inward_motion_seen
                    and now - last_two_hands_seen_at
                    <= CLAP_MISSING_TOLERANCE_SECONDS
                    and last_two_hand_gap <= 0.30
                ):
                    # 两手接触重叠时常会被识别成一只手；在中央位置推断击掌完成。
                    single_center = amplify_point(
                        palm_center(result.hand_landmarks[0]), args.movement_gain
                    )
                    if 0.35 <= single_center[0] <= 0.65:
                        trajectories.clear()
                        active_points.clear()
                        trajectory.clear()
                        recording = False
                        grabbed_index = None
                        grab_original_points = []
                        was_grabbing = False
                        clap_armed = False
                        clap_inward_motion_seen = False
                        last_clear_at = now
                        clap_cleared = True
                        status_message = "CLAP - canvas cleared"
                elif now - last_two_hands_seen_at > CLAP_MISSING_TOLERANCE_SECONDS:
                    # 快速击掌可能短暂漏掉一只手；持续丢失后才重置。
                    previous_hand_x = None
                    clap_armed = False
                    clap_inward_motion_seen = False
                    last_two_hand_gap = float("inf")

                if hand_visible and system_enabled:
                    landmarks = result.hand_landmarks[0]
                    thumb = landmarks[THUMB_TIP]
                    index = landmarks[INDEX_FINGER_TIP]
                    palm_width = distance(
                        landmarks[INDEX_FINGER_MCP], landmarks[PINKY_MCP]
                    )
                    fingertip_distance_px = math.hypot(
                        (thumb.x - index.x) * width,
                        (thumb.y - index.y) * height,
                    )
                    pinch_ratio = fingertip_distance_px / min(width, height)
                    contact_x, contact_y = amplify_point(
                        ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2),
                        args.movement_gain,
                    )
                    contact_px = (
                        round(contact_x * width),
                        round(contact_y * height),
                    )
                    hand_center = amplify_point(
                        palm_center(landmarks), args.movement_gain
                    )
                    grabbing = is_five_finger_grab(landmarks, palm_width)

                    # 握拳开始时，选中手掌附近的一条轨迹。
                    if (
                        not clap_cleared
                        and not delete_cooldown_active
                        and grabbing
                        and not was_grabbing
                        and not recording
                    ):
                        grabbed_index = nearest_trajectory(
                            trajectories, hand_center, width, height
                        )
                        if grabbed_index is not None:
                            grab_original_points = trajectories[grabbed_index].points[:]
                            grab_started_at = hand_center
                            grab_hand_size = max(palm_width, 1e-6)
                            status_message = "GRABBED - move or resize"

                    # 被抓取轨迹跟随手掌平移，并随画面中的手掌尺寸同步缩放。
                    if not clap_cleared and grabbing and grabbed_index is not None:
                        scale = palm_width / grab_hand_size
                        origin_x, origin_y = grab_started_at
                        moved_points = []
                        for point_x, point_y in grab_original_points:
                            moved_points.append(
                                (
                                    hand_center[0] + (point_x - origin_x) * scale,
                                    hand_center[1] + (point_y - origin_y) * scale,
                                )
                            )
                        trajectories[grabbed_index].points = moved_points
                        status_message = (
                            "Release in corner to delete"
                            if in_delete_corner(hand_center, width, height)
                            else "GRABBED - move or resize"
                        )

                    # 松手时固定对象；若手掌位于任一角落，则删除对象。
                    if (
                        not clap_cleared
                        and not grabbing
                        and was_grabbing
                        and grabbed_index is not None
                    ):
                        last_grab_release_at = now
                        if in_delete_corner(hand_center, width, height):
                            trajectories.pop(grabbed_index)
                            last_delete_at = now
                            status_message = (
                                f"Trajectory deleted - cooldown "
                                f"{DELETE_COOLDOWN_SECONDS:.1f}s"
                            )
                        else:
                            status_message = "Trajectory placed"
                        grabbed_index = None
                        grab_original_points = []

                    if (
                        not grabbing
                        and not was_grabbing
                        and grabbed_index is None
                        and not recording
                        and not clap_cleared
                        and not delete_cooldown_active
                        and not grab_cooldown_active
                        and pinch_ratio <= args.pinch_threshold
                    ):
                        recording = True
                        trajectory = []
                        active_points = []
                        recording_started_at = now
                        status_message = "RECORDING"

                    if recording and not clap_cleared:
                        if pinch_ratio >= args.release_threshold:
                            recording = False
                            if active_points:
                                trajectories.append(Trajectory(active_points[:]))
                            active_points = []
                            # 暂时不保存 CSV；需要恢复时取消下一行注释。
                            # saved_path = save_trajectory(trajectory, args.output_dir)
                            status_message = "Trajectory complete"
                        else:
                            trajectory.append(
                                (now - recording_started_at, contact_x, contact_y)
                            )
                            active_points.append((contact_x, contact_y))

                    # 只显示拇指和食指的中心点：默认红色，捏合时绿色。
                    point_color = (0, 255, 0) if recording else (0, 0, 255)
                    cv2.circle(canvas, contact_px, 9, point_color, -1)
                    was_grabbing = grabbing

                    cv2.putText(
                        canvas,
                        f"Pinch/screen: {pinch_ratio:.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        ui_color,
                        2,
                        cv2.LINE_AA,
                    )
                elif recording and system_enabled:
                    # 手短暂离开画面时结束当前记录，避免轨迹一直处于录制状态。
                    recording = False
                    if active_points:
                        trajectories.append(Trajectory(active_points[:]))
                    active_points = []
                    # 暂时不保存 CSV；需要恢复时取消下一行注释。
                    # saved_path = save_trajectory(trajectory, args.output_dir)
                    status_message = "Hand lost, trajectory complete"
                elif not hand_visible or not system_enabled:
                    was_grabbing = False
                    grabbed_index = None

                # 绘制所有已完成轨迹，以及当前正在创建的轨迹。
                paths = [item.points for item in trajectories]
                if active_points:
                    paths.append(active_points)
                for path in paths:
                    pixel_points = [
                        (round(x * width), round(y * height)) for x, y in path
                    ]
                    for start, end in zip(pixel_points, pixel_points[1:]):
                        cv2.line(canvas, start, end, (0, 0, 255), 4, cv2.LINE_AA)

                if grabbed_index is not None:
                    draw_delete_zones(canvas)

                if detected_hand_count == 0:
                    hand_hint = "Hands: 0 | Show one hand"
                elif detected_hand_count == 1:
                    hand_hint = (
                        "Hands: 1 | R -> L: system OFF"
                        if system_enabled
                        else "Hands: 1 | L -> R: system ON"
                    )
                else:
                    hand_hint = (
                        "Hands: 2 | Clap: clear canvas"
                        if system_enabled
                        else "Hands: 2 | Use one hand to start"
                    )

                status_color = (
                    (0, 0, 255)
                    if not system_enabled or recording
                    else (0, 180, 0)
                )
                cv2.putText(
                    canvas,
                    status_message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    hand_hint,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    ui_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    (
                        "System ON - swipe right to left to stop"
                        if system_enabled
                        else "System OFF - swipe left to right to start"
                    ),
                    (10, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    ui_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "Press Q or ESC to quit",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    ui_color,
                    2,
                    cv2.LINE_AA,
                )
                window_title = (
                    "Pinch Trajectory - AR"
                    if args.mode == "AR"
                    else "Pinch Trajectory - Whiteboard"
                )
                cv2.imshow(window_title, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    # 暂时不保存 CSV；需要恢复时取消下面两行注释。
                    # if recording:
                    #     save_trajectory(trajectory, args.output_dir)
                    break
    finally:
        camera_stream.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    if not 0 < args.pinch_threshold < args.release_threshold:
        print(
            "阈值必须满足：0 < --pinch-threshold < --release-threshold",
            file=sys.stderr,
        )
        return 2
    if args.movement_gain <= 0:
        print("--movement-gain 必须大于 0", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

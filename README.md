# Gesture

基于 OpenCV 和 MediaPipe 的实时手势绘图实验。项目可以通过摄像头追踪手部关键点，并使用拇指与食指捏合在白板或实时视频上绘制轨迹。

## 功能

- 实时手部关键点检测
- 拇指与食指捏合绘制
- 默认使用独立白板画布
- AR 模式下在摄像头视频上叠加绘制
- 保留多条独立轨迹
- 握拳抓取、移动和缩放轨迹
- 将轨迹拖至画面角落后松手删除
- 两只手从左右向中央击掌清屏

## 环境要求

- Python 3.11 或更高版本
- 可用的摄像头
- macOS、Windows 或 Linux

安装依赖：

```bash
python -m pip install -r requirements.txt
```

项目使用 MediaPipe Hand Landmarker 模型，默认路径为：

```text
models/hand_landmarker.task
```

如果模型文件不存在，可以下载官方模型：

```bash
mkdir -p models
curl -fL \
  -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 运行手部关键点检测

```bash
python main.py
```

该脚本在摄像头画面上绘制手部骨架、左右手信息和检测置信度。

如果默认摄像头不可用，可以指定其他摄像头编号：

```bash
python main.py --camera 1
```

## 运行手势画板

### 白板模式（默认）

```bash
python pinch_tracker.py
```

摄像头在后台用于识别手势，窗口只显示白色画布和绘制结果。

### AR 模式

```bash
python pinch_tracker.py --mode AR
```

AR 模式会将中心点、轨迹和操作提示叠加在实时摄像头画面上。模式名称不区分大小写。

## 手势操作

| 操作 | 手势 |
| --- | --- |
| 移动指针 | 移动拇指与食指，指针位于两指指尖中心 |
| 开始绘制 | 捏合拇指与食指，中心点由红色变为绿色 |
| 结束绘制 | 分开拇指与食指 |
| 抓取轨迹 | 将手掌移至轨迹附近并握拳 |
| 移动轨迹 | 保持握拳并移动手掌 |
| 缩放轨迹 | 保持握拳并让手靠近或远离摄像头 |
| 删除轨迹 | 将抓取的轨迹移至任一角落，然后张开手 |
| 清空画布 | 两手分别从画面左右向中央移动并击掌 |

删除轨迹后有 1.5 秒操作冷却时间。抓取轨迹时，手掌中心需要位于轨迹约 80 像素范围内。

## 调整捏合灵敏度

捏合比例的计算方式为：

```text
两指指尖像素距离 / 画面短边像素数
```

默认开始阈值为 `0.04`，分离阈值为 `0.06`：

```bash
python pinch_tracker.py \
  --pinch-threshold 0.04 \
  --release-threshold 0.06
```

增大阈值会让捏合更容易触发。分离阈值必须大于开始阈值，以避免临界距离造成状态抖动。

## 常用参数

```text
--camera CAMERA                 摄像头编号
--mode {WHITEBOARD,AR}          显示模式
--model MODEL                   MediaPipe 模型路径
--pinch-threshold VALUE         开始捏合阈值
--release-threshold VALUE       结束捏合阈值
```

查看全部参数：

```bash
python pinch_tracker.py --help
```

## 退出

在显示窗口中按 `Q` 或 `Esc`。

## 数据保存

代码中保留了将轨迹写入 CSV 的实现，但相关调用目前已注释，运行时不会创建 CSV 文件。CSV 功能可以在 `pinch_tracker.py` 中重新启用。

## macOS 摄像头权限

首次运行时，macOS 可能会请求摄像头权限。如果无法打开摄像头，请在“系统设置 → 隐私与安全性 → 摄像头”中允许终端或所使用的 IDE 访问摄像头。

# Stereo_Vision — Binocular VR Video Service

Remote HCR 双目视觉服务，支持 ZED / USB 双目相机 + WebRTC 低延迟视频流 + 深度/障碍检测。

---

## 快速开始

### Windows PowerShell

```powershell
# 1. 进入项目目录
cd E:\Remote_HCR\Stereo_Vision

# 2. 创建虚拟环境（首次）
python -m venv .venv

# 3. 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 4. 安装依赖（首次）
python -m pip install aiohttp opencv-python numpy aiortc av jinja2 aiofiles

# 5. 生成 HTTPS 证书（首次）
mkdir config -ErrorAction SilentlyContinue
& "C:\Program Files\Git\usr\bin\openssl.exe" req -x509 -nodes -days 365 `
    -newkey rsa:2048 `
    -keyout config\key.pem `
    -out config\cert.pem `
    -subj "/CN=localhost/O=StereoVision"

# 6. 启动服务
python main.py
```

服务启动后访问：**`https://localhost:9000/`**

> 证书不存在时服务自动降级为 HTTP（打印警告），不影响使用。

### 非首次使用

```powershell
# 方式一：直接运行（无需激活环境）
cd E:\Remote_HCR\Stereo_Vision
.\.venv\Scripts\python.exe main.py

# 方式二：激活环境后运行
cd E:\Remote_HCR\Stereo_Vision
.\.venv\Scripts\Activate.ps1
python main.py
```

### Linux / macOS

```bash
# 首次
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash ./config/gen_self_signed_cert.sh

# 每次启动
source .venv/bin/activate
python main.py
```

### 停止服务

```powershell
# 方法一：Ctrl+C（前台运行时）

# 方法二：强制结束所有 Python 进程
taskkill /F /IM python.exe

# 方法三：只结束服务进程（推荐）
# 先查端口
netstat -ano | findstr ":9000.*LISTENING"
# 输出中的最后一列是 PID，例如 12345
taskkill /F /PID 12345
```

> ⚠️ 调试时摄像头可能被其他 Python 进程占用。启动新服务前，先用 `taskkill /F /IM python.exe` 释放摄像头资源。

---

## 相机选择与配置

### 环境变量说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `STEREO_CAMERA_TYPE` | `zed` | 选 `zed` 或 `usb` |
| `STEREO_USB_LEFT_INDEX` | `2` | 左摄像头设备索引 |
| `STEREO_USB_RIGHT_INDEX` | `2` | 右摄像头设备索引（同索引=单设备SBS模式） |
| `STEREO_USB_WIDTH` | `640` | 目标宽度（像素） |
| `STEREO_USB_HEIGHT` | `480` | 目标高度（像素） |
| `STEREO_USB_FPS` | `15` | 帧率 |
| `STEREO_USB_SCALE` | `0.5` | 立体校正缩放比例 |
| `STEREO_PORT` | `9000` | HTTP/HTTPS 端口 |

### ZED 相机模式（默认）

```powershell
$env:STEREO_CAMERA_TYPE = "zed"
python main.py
```

### USB 双目摄像头模式

```powershell
$env:STEREO_CAMERA_TYPE = "usb"
python main.py
```

> USB 模式下深度由 OpenCV SGBM 立体匹配算法估计，无需 ZED SDK。精度低于 ZED，但无需额外硬件。

---

## 🔧 USB 双目摄像头调试指南

### 摄像头 Index 不固定

> ⚠️ **最重要的一条**：Windows 上 USB 摄像头每次插拔索引都可能变化，**每次调试前必须重新扫描**。

### 第一步：扫描所有摄像头

摄像头索引在 Windows 上不是固定的——每次插拔都可能导致索引变化。进入项目目录执行：

```powershell
python scan_cameras.py
```

典型输出：

```
Index 0: 640x480  → 电脑摄像头
Index 1: 1280x720 → 手机摄像头（App 模式）
Index 2: 640x480  → 双目摄像头（或电脑摄像头）
```

记下双目摄像头对应的索引，然后进入下一步。

### 第二步：验证摄像头

逐一测试每个摄像头，看哪个是你的双目：

```powershell
# 改 --index 测试每个
python test_camera_select.py --index 0
python test_camera_select.py --index 1
python test_camera_select.py --index 2
```

用浏览器打开 `http://localhost:9000/` 确认画面。

### 第三步：验证 SBS 格式

SBS（Side-by-Side）格式下，左右半边应该是**完全不同的画面**。用手挡住左镜头，左边应该变黑；挡住右镜头，右边应该变黑。

```python
# 验证脚本（verify_sbs.py）
import cv2, time

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
time.sleep(0.3)

for _ in range(3):
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    eye_w = w // 2
    left = frame[:, :eye_w]
    right = frame[:, eye_w:]
    diff = abs(left.astype(int) - right.astype(int)).mean()
    status = "OK: 左右不同" if diff > 5 else "ERROR: 左右相同"
    print(f"Frame: {w}x{h}, diff={diff:.1f} {status}")

cap.release()
```

输出 `diff > 5` 说明左右是不同画面。

### 第四步：确认分辨率

> 这是最常踩的坑。

普通 USB 摄像头和 USB 双目摄像头的分辨率含义完全不同：

| 摄像头类型 | 分辨率 | 说明 |
|-----------|--------|------|
| 普通 USB 摄像头 | 640×480 | 单画面 |
| **USB 双目（SBS 模式）** | **1280×480** | **左右各 640 像素拼接** |
| 手机摄像头（App） | 1280×720 | 手机 App 输出 |

如果分辨率设错，双目摄像头会回退到单画面模式。

```python
# 逐个试分辨率，看哪个左右不同
import cv2, time

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

for w, h in [(640,480), (1280,480), (1280,720), (2560,480)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(0.3)
    ret, frame = cap.read()
    if not ret:
        continue
    eye_w = frame.shape[1] // 2
    diff = abs(frame[:,:eye_w].astype(int) - frame[:,eye_w:].astype(int)).mean()
    status = "SBS OK" if diff > 5 else "单画面"
    print(f"Set {w}x{h} -> Got {frame.shape[1]}x{frame.shape[0]}, diff={diff:.1f} [{status}]")

cap.release()
```

### 快速诊断清单

下次出问题，按这个顺序排查：

1. **摄像头 Index 对了吗？** → `python scan_cameras.py`
2. **分辨率对了吗？** → 用上面的脚本测试不同分辨率
3. **左右真的不同吗？** → `python verify_sbs.py`
4. **AMCap SBS 模式开了吗？** → AMCap 里要选择 Side-by-Side 输出格式

### 常见问题

**Q: 两边画面完全相同？**

最可能原因：分辨率设置错误。双目摄像头要设 1280×480，不能设 640×480。参见上面的分辨率测试脚本。

**Q: AMCap 里能看到两个画面，但 OpenCV 只能读到一个？**

AMCap 通过 DirectShow 能访问双 pin，但 OpenCV 默认只抓第一个 pin。解决方案是找到摄像头在 SBS 模式下的正确分辨率（通常是 1280×480），让摄像头输出真正的 SBS 格式。

**Q: 用手机做双目摄像头，左右还是一样？**

手机 App（USB Camera、IP Camera 等）需要在 App 里切换到 **Side-by-Side 模式**，App 预览里看到两个画面并排后，OpenCV 才能正确读取。

---

## 页面说明

| 路径 | 说明 |
|------|------|
| `/` | 首页入口 |
| `/hud` | 视频 + 顶部 HUD 控制面板（实时调参，推荐） |
| `/stereo?transport=webrtc` | WebRTC 低延迟双目（推荐，需 TURN 服务器） |
| `/depth` | 深度可视化（彩色深度图 + 统计数据） |
| `/control` | 显示参数控制台（裁切/畸变/分隔线） |
| `/debug` | 调试：FPS + 原始帧 |
| `/video_feed` | MJPEG 直接流（双目） |
| `/depth/feed` | MJPEG 深度彩图流 |

## API Endpoints

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/params` | GET/POST | 获取/更新显示参数 |
| `/api/params/reset` | POST | 重置为默认值 |
| `/api/status` | GET | 相机连接状态 |
| `/api/depth/stats` | GET | 深度统计（min/max/mean/coverage） |
| `/api/depth/roi` | GET | 指定 ROI 深度统计 |
| `/api/webrtc/config` | GET | WebRTC ICE 配置 |
| `/api/obstacle/status` | GET | 障碍检测结果 |

## 显示参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| `crop` | 0.5–1.0 | 每只眼睛裁切比例 |
| `xshift` | -80–80 | 瞳距微调（像素） |
| `k1` | -0.5–0.5 | 桶形畸变系数 |
| `k2` | -0.5–0.5 | 二次畸变系数 |
| `sep` | 0–200 | 左右眼中间黑边（像素） |
| `gshift_x` | -200–200 | 整体水平平移 |
| `gshift_y` | -200–200 | 整体垂直平移 |

## WebRTC 配置

WebRTC 需要 TURN 服务器才能在外网环境正常工作（STUN 服务器只能获取公网 IP，无法穿透防火墙）。

本地调试用 HTTP 流即可，无需 TURN。

配置 TURN 服务器，在 `config.py` 中设置：

```python
STEREO_TURN_URLS = "turn:your-turn-server.com:3478"
STEREO_TURN_USERNAME = "your-username"
STEREO_TURN_PASSWORD = "your-password"
```

---

## 目录结构

```
Stereo_Vision/
├── main.py                  # 服务入口
├── config.py                # 配置与环境变量
├── state.py                 # app[] 键定义
├── camera/                  # 相机抽象层
│   ├── stereo_camera_base.py
│   ├── zed_camera.py        # ZED SDK 后端
│   └── usb_camera.py        # USB 双目后端
├── processing/              # 图像算法
│   ├── display.py           # 畸变校正/裁切
│   ├── depth_processor.py   # SGBM 深度估计
│   └── obstacle_detector.py
├── transport/               # 传输层
│   ├── mjpeg_stream.py      # MJPEG 流
│   └── webrtc_tracks.py     # WebRTC VideoTrack
├── services/                # 业务层 + 持久化
├── routers/                 # HTTP/WS 路由
├── templates/               # HTML 页面
├── static/                  # CSS + JS
├── config/                  # HTTPS 证书
├── scripts/                 # 启动脚本
└── docs/
```

## 与 Client_RHCR 的关系

- `Stereo_Vision :9000` — 双目视频 + 深度 + VR
- `Client_RHCR :8000` — 单目 AprilTag 位姿 + 夹爪

两服务并行部署；在 Client 设置页链到 `https://<ip>:9000/` 即可。

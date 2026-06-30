# 02 · 阶段 0–4 落地计划

本文件记录本次代码改动的**模块、接口、依赖、验证**,与计划书 `cup_stereo_depth.plan.md` 一一对应。

---

## 阶段 0 · 硬件常量 + 立体校正

### 0.1 硬件常量
- 新建 `config/hardware.py`(统一源)。
- `BASE_DIR = Path(__file__).resolve().parent.parent` — 仓库根。
- `CALIB_NPZ_PATH = BASE_DIR / "config" / "calibration" / "calib.npz"` — 外部产出。
- 公式函数 `z_cm_from_disparity(disp)`。
- 等比例 helper:`resize_for_sgbm` / `resize_to_mono`。

### 0.2 双目立体校正器
- 新建 `camera/calibration.py::StereoCalibrator`。
- 优先读 `map1_l, map2_l, map1_r, map2_r`(现成 remap 表);无则用 `K_l, D_l, K_r, D_r, R, T` + `cv2.stereoRectify` + `cv2.initUndistortRectifyMap` 现场算。
- 降级模式:无 npz 时 `is_rectified = False`,`rectify()` 直通返回输入(警告 + 精度变差)。
- 调试方法 `verify_vertical_parity(left_rect, right_rect)` — 极线对齐度量(越小越好)。

### 0.3 USB 摄像头改造
- 改造 `camera/usb_camera.py`:
  - `__init__` 末尾加载 `StereoCalibrator`(失败不崩,降级)。
  - 删除原来的 identity map `_remap`,改名为 `_apply_rectify`,内部走 calibrator。
  - **新增** `read_rectified_pair() -> (left, right) | None` — 阶段 2 SGBM 用。
  - `read_stereo()` 走已校正路径,`cv2.hconcat` 出 (SBS_H, SBS_W, 3),**不**再做 `_resize_if_needed`。
  - `read_depth()` 走已校正路径,内部调 `_compute_depth`。
  - 删除 `_resize_if_needed`(单/双设备模式最终都是 2*target_w × target_h,不再需要 resize)。

### 0.4 验证
- 无 npz → 警告 + 降级直通(不崩)。
- 内参+外参形式 npz → 加载 + stereoRectify + 算 remap 成功。
- 现成 map 形式 npz → 直接加载。
- 极线对齐度量可用(参数真实值会低)。
- `read_stereo` 输出形状 = (SBS_HEIGHT, SBS_WIDTH, 3) = (1080, 3840, 3)。

---

## 阶段 1 · YOLOv8 cup 检测(单例,锁 ID=41)

### 1.1 检测器
- 新建 `processing/detector.py::CupDetector`。
- 加载:`YOLO(str(MODELS_DIR/"yolov8s.pt")) if exists else "yolov8s.pt"`,自动下载回退。
- 类别锁:`classes=[TARGET_CLS_ID] = [41]`。
- 置信度:`conf=DEFAULT_CONF_THRESHOLD = 0.5`。
- imgsz:默认 640(实测 CPU 1080p 上 ~80ms/帧 ≈ 12fps;`imgsz=320` ~30fps,漏检风险高)。
- 后处理双保险:再按 `cls == 41` 过滤 + 按面积最大。
- 边界:`x1, y1, x2, y2` 夹紧到画面范围。
- 无 cup → `(None, None)`。

### 1.2 验证
- 加载 1 次,后续推理不再加载(查 `_model` id 一致)。
- 空白图 → `(None, None)`。
- 假文字 "cup" → 仍 `(None, None)`(YOLO 不是 OCR)。
- 性能:CPU 上 1920×1080 推理 < 100ms/帧。

---

## 阶段 2 · SGBM 视差 + 公式 + 滑动均值

### 2.1 解算器
- 新建 `processing/stereo_depth.py::StereoDepthSolver`。
- SGBM 参数:`numDisparities=64`, `blockSize=5`, P1/P2 用 OpenCV 公式。
- 视差图缩放:**关键修正** — 缩到 sgbm 分辨率算视差后,反推回 mono 坐标时
  **视差值乘以 1/scale**(因为 sgbm 视差 = mono 视差 × scale)。
- `read_depth_at(cx, cy)` 在 mono 视差图上采样,公式 → 钳位 → round 1 位。
- 滑动窗口:`collections.deque(maxlen=DEPTH_SMOOTH_WINDOW)`,`smoothed_depth` 返回均值。
- 公式哨兵:模块顶部 `assert FOCAL_LENGTH_MM == 3.0, BASELINE_CM == 6.0`。

### 2.2 验证
- 公式自检:`z_cm_from_disparity(5.0) = 3.6`(任务规约字面)。
- 合成图测试:60px 视差 → 测得 ~64(SGBM 误差 7%);15px 视差 → 测得 1.0cm(字面公式 1.2,误差 0.2cm)。
- 滑动窗口:`smoothed_depth([3.6, 3.7, 3.5, 3.8, 3.6, 3.7])` 推第 6 次后均值 = 3.66 ≈ 3.7。
- 越界 `cx<0/cy>=H` → None。
- 无 compute 缓存 → None。

---

## 阶段 3 · 三色分级 + 1Hz 闪烁

### 3.1 渲染器
- 新建 `processing/warning.py::WarningOverlay`。
- 等级判定(任务规约字面):
  - `depth > SAFE_CM (5)` → safe(绿框,厚 2)
  - `DANGER_CM (2) <= depth <= SAFE_CM` → work(黄框,厚 2)
  - `0 < depth < DANGER_CM` → danger(红框,厚 5,1Hz 闪烁)
  - `depth is None` → none(灰框,厚 1,文字"无有效深度")
- 距离文字(中文):`相机至目标距离:XX.X cm`,左上角白底黑字,PIL 渲染(CJK 字体优先级 simhei → NotoSansSC → simfang → msyh)。
- 顶部 DANGER 横幅:中文"危险 过近",闪烁时 65% 半透明叠加。
- 左右眼用同一 box / 同一 depth 渲染 → VR 对称无重影。

### 3.2 验证
- 等级判定边界(>5 safe, =5 work, <2 danger, None none)。
- 闪烁相位:跨相位 20 次采样约 10 True(50%)。
- 中文渲染:simhei.ttf 加载 + 中文显示无乱码。
- 错误输入 `safe <= danger` → ValueError。

---

## 阶段 4 · SBS 3840×1080 单帧总入口

### 4.1 管线
- 新建 `processing/sbs_pipeline.py::SBSPipeline`。
- 构造注入:`camera`, `calibrator`, `detector`, `solver`, `warn`。
- `process_one_frame()`:
  1. `camera.read_rectified_pair()`(已校正)
  2. YOLO **只对左眼** 跑一次
  3. SGBM 算视差
  4. 在 cup 中心读深度 + 滑动均值
  5. 渲染(左右眼对称)
  6. `cv2.hconcat([left, right])` → 严格 (1080, 3840, 3)
- 异常隔离:任何子步骤失败,继续后续(不让管线挂掉),只记录 warning。

### 4.2 应用层 wiring
- `state.py` 新增 `PIPELINE_KEY = "stereo_pipeline"`。
- `wiring/lifecycle.py` 启动期:
  - 打开摄像头后,组装 `SBSPipeline` 并写入 `app[PIPELINE_KEY]`。
  - 失败时回退,`PIPELINE_KEY = None`(路由层走 `cam.read_stereo()` 兜底)。
- `routers/stream_router.py`:
  - `_build_frame_fn(app)` — 优先 pipeline,降级到 cam,都没有时返回 `lambda: None`。
  - **不**再调 `DisplayProcessor`(本任务硬约束"无拉伸、无裁切")。
  - WebRTC `fallback_shape = (SBS_HEIGHT, SBS_WIDTH, 3) = (1080, 3840, 3)`。
  - MJPEG 用 `USB_FPS=10` 限速。

### 4.3 验证
- 整体 `from main import create_app` 链路 import 无错。
- 模拟 5 帧端到端测试:SBS 形状全部 (1080, 3840, 3);danger 帧渲染红框 + 中文横幅 + 距离文字 1.0cm。
- 帧延迟 ~160ms(CPU 推理 + SGBM),实际部署可用 GPU 或减小 imgsz 提速。

---

## 依赖与安装

`requirements.txt` 新增:
```
ultralytics>=8.0.0
ultralytics-thop>=2.0.0
```

安装命令(用清华源加速):
```bash
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics ultralytics-thop
```

模型权重 `yolov8s.pt`:
- 优先用 `models/yolov8s.pt`(本地,避免每次联网)
- 缺失时 `CupDetector.__init__` 自动从 ultralytics GitHub release 下载到当前目录(~22MB)

---

## 文件变更清单

| 状态 | 路径 | 说明 |
|---|---|---|
| 新增 | `config/hardware.py` | 硬件常量、阈值、COCO ID、SGBM 参数、公式、等比例 helper |
| 新增 | `config/__init__.py` | 包初始化(原 `config.py` 内容整合进来) |
| 新增 | `config/calibration/.gitkeep` | 占位,等外部 npz |
| 删除 | `config.py`(根) | 内容已合并到 `config/__init__.py` |
| 新增 | `camera/calibration.py` | `StereoCalibrator`(npz 加载 + 校正) |
| 改造 | `camera/usb_camera.py` | 接入 calibrator,新增 `read_rectified_pair`,删除 identity remap |
| 新增 | `processing/detector.py` | `CupDetector`(YOLOv8s 单例) |
| 新增 | `processing/stereo_depth.py` | `StereoDepthSolver`(SGBM + 公式 + 滑动均值) |
| 新增 | `processing/warning.py` | `WarningOverlay`(三色 + 闪烁 + 中文) |
| 新增 | `processing/sbs_pipeline.py` | `SBSPipeline`(一帧总入口) |
| 改造 | `state.py` | 新增 `PIPELINE_KEY` |
| 改造 | `wiring/lifecycle.py` | 启动期组装 SBSPipeline |
| 改造 | `routers/stream_router.py` | `frame_fn` 切到 pipeline,fallback_shape 改 SBS |
| 改造 | `requirements.txt` | 加 ultralytics + ultralytics-thop |
| 新增 | `docs/01_development_scope.md` | 本次任务的功能边界 |
| 新增 | `docs/02_stage_plan.md` | 本文件 |

---

## 实施顺序(已按此执行)

1. **环境核查** — `.venv` 装 ultralytics + ultralytics-thop,验证 `yolov8s.pt` 可加载 ✅
2. `config/hardware.py` ✅
3. `camera/calibration.py` + 改造 `usb_camera.py` ✅
4. `processing/detector.py` ✅
5. `processing/stereo_depth.py` ✅
6. `processing/warning.py` ✅
7. `processing/sbs_pipeline.py` + 改造 `wiring/lifecycle.py` + 改造 `routers/stream_router.py` ✅
8. `requirements.txt` 加 ultralytics ✅
9. 文档 `docs/01_*.md`、`docs/02_*.md` ✅

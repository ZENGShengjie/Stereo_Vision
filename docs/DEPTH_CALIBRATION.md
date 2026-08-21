# 深度校准完整指南

本指南说明如何通过 ChArUco 标定 + DISP_SCALE 系统性校准 + 分段线性插值，将深度测量精度从"粗估"提升到"厘米级准确"。

---

## 一、ChArUco 极线校正（最根本的精度保障）

### 1.1 为什么需要极线校正？

未经校正的双目图像，左右相机的同名点不在同一扫描线上。SGBM 需要在 2D 范围内搜索匹配点，精度差且速度慢。

极线校正后，左右图像的同名点严格对齐到同一水平扫描线，SGBM 从 2D 搜索降为 1D 极线搜索，匹配精度大幅提升。

### 1.2 标定板制作

使用 ChArUco 标定板（比传统棋盘格更鲁棒，ArUco 角点自动 sub-pixel refine，部分遮挡也可用）：

- **规格**：7 列 × 5 行 ArUco，DICT_6X6_250
- **方格尺寸**：Square = 40.57 mm
- **ArUco 尺寸**：Marker = 29.95 mm（典型为方格的 0.7~0.8 倍）

> ⚠️ **必须用数显游标卡尺实测打印件**！打印机的缩放比例会导致实际尺寸和标称值不符，必须实测后更新 `config/hardware.py` 中的 `CHARUCO_SQUARE_MM` 和 `CHARUCO_MARKER_MM`。

```python
# config/hardware.py — 打印后用卡尺实测这两个值
CHARUCO_SQUARE_MM: float = 40.57   # ← 实测值，可能需要调整
CHARUCO_MARKER_MM: float = 29.95  # ← 实测值，可能需要调整
```

### 1.3 采集标定图像

1. 安装依赖：
   ```powershell
   pip install opencv-contrib-python==4.10.0.84
   ```

2. 运行采集工具：
   ```powershell
   cd E:\Remote_HCR\Stereo_Vision
   python scripts/calibrate_stereo.py
   ```

3. 采集要求：
   - 标定板必须**同时**出现在左右两个画面中
   - 采集 15-30 对，覆盖各种角度与距离（近/远、左右倾、上下倾、居中/边缘）
   - 按 **SPACE** 接受一对，按 **R** 撤回一对，按 **ESC** 结束采集

4. 采集完成后自动保存到 `config/calibration/calib.npz`

### 1.4 标定流程说明

`scripts/calibrate_stereo.py` 执行以下步骤：

```
Step 1: cv2.stereoCalibrate (SAME_FOCAL_LENGTH 约束)
         ↓
Step 2: cv2.stereoCalibrate (FIX_INTRINSIC 精调)
         ↓
Step 3: cv2.calibrateCamera 单眼细化 (左/右各自)
         ↓
Step 4: cv2.stereoCalibrate 最终 (固定内参)
         ↓
Step 5: cv2.stereoRectify 计算 R1/R2/P1/P2/Q
         ↓
Step 6: cv2.initUndistortRectifyMap 生成 remap 映射表
         ↓
保存到 config/calibration/calib.npz
```

---

## 二、DISP_SCALE 系统性校准（精度提升 10-30%）

### 2.1 问题根源

深度公式：

```
Z_cm = (f_px × BASELINE_CM) / (d × DISP_SCALE)
```

`f_px` 从 `HFOV=80°` 反推，是估算值，与摄像头真实像素焦距存在系统性偏差（典型 1.5-4x）。

### 2.2 校准方法

打开 `/perf` 页面，把目标放在已知距离（如 25cm），用尺子量出真实距离，计算：

```
DISP_SCALE = 真实距离(cm) / 当前显示距离(cm)
```

例如：真实距离 25cm，显示 22.5cm → `DISP_SCALE = 25 / 22.5 = 1.1111`

### 2.3 在线校准

在 `/perf` 页面点击"焦距校准 25 cm"按钮，或通过 API：

```bash
curl -X POST http://localhost:9000/api/calibrate/disp_scale \
  -H "Content-Type: application/json" \
  -d '{"real_cm": 25.0, "z_measured_cm": 22.5}'
```

校准值会保存到 `data/hardware_overrides.json`（不提交到 Git）。

---

## 三、分段线性插值（4 点校准）

在 EMA 平滑之后，系统还会做一次分段线性插值，消除非线性误差。

内置 4 个校准点：

```python
# processing/stereo_depth.py
self._calib_points: list[tuple[float, float]] = [
    (11.2, 11.5),   # (传感器读数, 真实距离)
    (12.4, 13.5),
    (13.9, 15.5),
    (15.6, 17.5),
]
```

在 `/perf` 页面可以动态添加/删除校准点。添加 2 个以上点后自动生效。

---

## 四、精度影响因子汇总

| 因子 | 当前状态 | 影响程度 | 解决方案 |
|------|----------|----------|----------|
| 极线校正 | ✅ 已做（calib.npz） | 🔴 20-50% | 运行 ChArUco 标定 |
| DISP_SCALE | ✅ 已校准（=1.1111） | 🔴 10-30% | /perf 页面实测校准 |
| 焦距估算 | 用 HFOV 反推 | 🟡 5-15% | 标定后可从 K_l[0,0] 读取 |
| SGBM 分辨率 | 320×240（降采样） | 🟡 3-8% | 可尝试 640×480 |
| 基线测量 | 硬编码 6.0cm | 🟡 1-5% | 标定后用真实 T 向量值 |
| EMA 平滑 | ✅ 已做（置信度自适应） | ✅ 良好 | 继续使用 |
| 直方图峰值 | ✅ 已做（抗杯柄/反光） | ✅ 良好 | 继续使用 |

---

## 五、注意事项

1. **calib.npz 是环境私有的**：每台相机的标定结果不同，标定文件不上传 GitHub
2. **硬件参数实时可调**：DISP_SCALE、BASELINE、HFOV 等都可以在 `/perf` 页面在线修改，无需重启
3. **校准历史可追溯**：`data/hardware_override_log.jsonl` 记录了每次参数变更

# 01 · 开发范围与硬件约定

本文件记录本仓库当前阶段(双目视觉头部深度测距·cup 替代)的**功能边界**和**硬件真理源**,
后续阶段如需扩展,先改本文档对齐规约,再改代码。

---

## 1. 任务一句话总结

把单设备 USB 双目摄像头(3840×1080 SBS 整帧) → **立体校正** → **YOLOv8 cup 检测**(只识别 COCO ID=41) → **SGBM 视差 + 公式深度** → **三色分级预警** → 拼回 3840×1080 SBS,作为 VR Box 的左右分屏输入流。

"cup 替代"的原因:人体头部(头发、衣物)纹理复杂、视差跳变严重,无法作为稳健参照物;用 cup 刚性曲面物体做参照,测距单调平滑。

---

## 2. 硬件全局常量(全部从 `config.hardware` 读,**禁止**在其它文件硬编码)

| 项 | 值 | 含义 |
|---|---|---|
| `FOCAL_LENGTH_MM` | `3.0` | 镜头物理焦距(mm) |
| `BASELINE_CM` | `6.0` | 双目基线(cm) |
| `HFOV_DEG` | `80.0` | 镜头水平视场角(°),用于反推像素焦距 |
| `MONO_WIDTH` × `MONO_HEIGHT` | `1920 × 1080` | 单目物理分辨率,锁死 |
| `SBS_WIDTH` × `SBS_HEIGHT` | `3840 × 1080` | 双目 SBS 整帧,锁死 |
| `SAFE_CM` | `5.0` | 安全接近区阈值(绿色框) |
| `DANGER_CM` | `2.0` | 危险接触区阈值(红色闪烁) |
| `DANGER_FLASH_HZ` | `1.0` | 危险区闪烁频率 |
| `DEPTH_SMOOTH_WINDOW` | `5` | 深度滑动均值窗口长度(防抖) |
| `TARGET_CLS_ID` | `41` | COCO cup 类别 ID |
| `DEFAULT_CONF_THRESHOLD` | `0.5` | YOLO 置信度阈值 |
| `SGBM_WIDTH` × `SGBM_HEIGHT` | `320 × 240` | SGBM 内部计算分辨率(等比例缩放,非锁定) |
| `SGBM_NUM_DISPARITIES` | `64` | SGBM 视差范围 |
| `SGBM_BLOCK_SIZE` | `5` | SGBM 匹配块大小 |

> **SGBM 分辨率不是锁死的**:由于 resize 硬约束(等比例、不裁切、不留黑边),
> 1920×1080(16:9)→ SGBM 时是 320×180(16:9),不是 320×240(4:3)。
> SGBM 在 320×180 上跑完全 OK,只是视场比预期小一截,不影响立体匹配。
> 真标定后 SGBM 视差精度足够。

---

## 3. 深度公式(任务规约字面)

```
Z_mm = (FOCAL_LENGTH_MM * BASELINE_CM * 10) / d
Z_cm = Z_mm / 10
```

- `d` 是"等效视差"(像素),`d > 0`。
- 全代码**只此一处**出现 `FOCAL_LENGTH_MM * BASELINE_CM` 乘积,在 :func:`config.hardware.z_cm_from_disparity`。
- 模块导入期有 `assert FOCAL_LENGTH_MM == 3.0, BASELINE_CM == 6.0` 哨兵,误改常量会启动失败。

> ⚠️ **物理说明**:该公式字面代入(3, 6, d=5)→ 3.6 cm,
> 与"15cm 起步测距"的常识不一致。**任务规约字面要求**就是如此,
> 实测距离有偏差时请优先调整:1) 标定 npz 的内参;2) 确认 SGBM 视差本身精度;
> 而**不要**改公式或物理常量。

---

## 4. resize 硬约束(代码层统一)

1. **只允许等比例**:`fx == fy`。
2. **单目锁死 1920×1080,SBS 锁死 3840×1080**;任何缩放只发生在 SGBM 内部。
3. **统一入口**:`config.hardware.resize_for_sgbm` / :func:`config.hardware.resize_to_mono`。
4. **不留黑边、不裁切**:用 `min(sx, sy)` 选缩放比。

---

## 5. 术语规范

- **相机到目标绝对深度 `Z`**(单位 cm) — 全文统一,不再用"头部/头皮/人脸"。
- **"目标"** = COCO ID=41 cup(等效头皮参照物),非真实人体头部。
- 变量命名:`target_depth_cm`、`depth_cm`、`smoothed_depth_cm`,**不要**用 `head_depth` / `face_distance` 这类旧名。

---

## 6. 本次实现范围(已做)

| 阶段 | 内容 | 文件 |
|---|---|---|
| 0 | 硬件常量 + npz 标定 + 立体校正器 | `config/hardware.py`, `camera/calibration.py`, `camera/usb_camera.py` |
| 1 | YOLOv8 cup 检测(单例,锁 ID=41) | `processing/detector.py` |
| 2 | SGBM 视差 + 公式 + 滑动均值 | `processing/stereo_depth.py` |
| 3 | 三色分级 + 1Hz 闪烁 + 中文距离文字 | `processing/warning.py` |
| 4 | 3840×1080 SBS 管线(单帧总入口) | `processing/sbs_pipeline.py`, `routers/stream_router.py`, `wiring/lifecycle.py` |

---

## 7. 全局排除(本次明确**不**实现)

- ❌ 棋盘格标定采集/标定生成代码(只读取外部产出的 `config/calibration/calib.npz`)
- ❌ ZED 摄像头分支(本任务锁定 USB)
- ❌ 十字准星、手眼标定
- ❌ 推子检测 / 推子数据集训练
- ❌ 独立工作区域框(cup 框天然替代)
- ❌ 推子与目标相对深度差
- ❌ 深度伪彩热力图叠加(`processing/depth_processor.py` 保持原样)
- ❌ 障碍检测(`processing/obstacle_detector.py` 保持原样)
- ❌ VR 视觉调参面板(`processing/display.py` + `models/display_params.py` 保持原样,本任务不调)

---

## 8. 验收对照(任务规约原话)

| 验收点 | 计划哪一步保证 |
|---|---|
| cup 校正后左右垂直偏移 < 1px | §0 `StereoCalibrator.rectify` 用 `cv2.remap INTER_LINEAR`;npz 必须来自正确标定流程 |
| cup 检测稳定、置信度 0.5、无 cup 返回空 | §1 `conf=0.5`,`classes=[41]`,取面积最大;空 → `target_depth_cm = None` |
| 15cm 推近 距离单调递减 | §2 滑动均值窗口长度 5;§2 公式硬约束;§3 渲染即时刷新 |
| 三色 + 闪烁 | §3 `WarningOverlay`;阈值在 `config/hardware.py` 可改 |
| 3840×1080 SBS 无拉伸、无裁切 | §4 `cv2.hconcat([left, right])` 无 resize;左右 box 同一变量 |

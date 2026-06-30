# Stereo_Vision API Reference

## REST Endpoints

### `GET /api/params`

返回当前显示参数。

```json
{
  "crop": 0.7,
  "xshift": 0,
  "k1": -0.18,
  "k2": 0.04,
  "sep": 40,
  "gshift_x": 0,
  "gshift_y": 0
}
```

### `POST /api/params`

更新显示参数（只更新提供的字段，实时生效，自动保存到 `data/display_params.json`）。

```json
{
  "xshift": 10,
  "sep": 60
}
```

### `POST /api/params/reset`

重置为默认值。

### `GET /api/status`

```json
{
  "zed": "open",
  "params": { ... }
}
```

`zed` 可选值：`open` | `not_available` | `error`

---

### `GET /api/depth/stats`

返回当前帧深度统计。

```json
{
  "min": 0.32,
  "max": 9.87,
  "mean": 2.14,
  "median": 1.95,
  "std": 1.03,
  "valid_px": 423100,
  "total_px": 442368,
  "coverage": 0.956
}
```

### `GET /api/depth/roi?x=&y=&w=&h=`

返回指定矩形 ROI 的深度统计。

### `GET /depth/feed`

深度图 MJPEG 流（彩色，colormap = TURBO）。

---

### `GET /api/obstacle/status`

返回障碍检测结果。

```json
{
  "clear": false,
  "obstacles": [
    {
      "col": 4,
      "row": 3,
      "x": 256,
      "y": 192,
      "w": 64,
      "h": 64,
      "min_depth": 0.31,
      "coverage": 0.42
    }
  ],
  "min_distance": 0.31,
  "grid_cols": 40,
  "grid_rows": 11
}
```

`clear: true` 表示路径畅通；`obstacles` 列出所有障碍格子。

### `GET /api/obstacle/params`

读取障碍检测阈值：

```json
{
  "cell_width": 64,
  "cell_height": 64,
  "min_confidence": 80,
  "depth_threshold": 0.5,
  "coverage_ratio": 0.3
}
```

### `POST /api/obstacle/params`

更新阈值（同上 JSON body）。

---

## Streaming

### `GET /video_feed`

MJPEG 双目视频流（经过显示参数处理后）。

### `GET /depth/feed`

MJPEG 深度彩图流。

---

## WebSocket Signaling

路径：`/ws/webrtc`

### 客户端 → 服务端

```json
{ "type": "webrtc_offer", "sdp": "..." }
```

### 服务端 → 客户端

连接就绪：

```json
{ "type": "signaling_ready" }
```

收到 offer 后返回：

```json
{ "type": "webrtc_answer", "sdp": "..." }
```

错误：

```json
{ "type": "error", "message": "..." }
```

### `GET /api/webrtc/config`

返回 WebRTC ICE 配置：

```json
{
  "iceServers": [
    { "urls": ["stun:stun.l.google.com:19302"] }
  ]
}
```

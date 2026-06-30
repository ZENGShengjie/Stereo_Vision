# Stereo_Vision 安装说明

## 系统要求

- Python 3.10 或 3.11
- ZED 相机（SK 或 ZED 2/2i/CMC）
- 网络：HTTPS/WSS（公网手机访问需要）

## 步骤 1 — 安装 Python 依赖

```bash
./scripts/install.sh
```

或手动：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 步骤 2 — ZED SDK

`pyzed.sl` **不在 PyPI**，需单独安装：

1. 下载 ZED SDK：https://www.stereolabs.com/developers/
2. 安装后，将 ZED Python API 加入 `PYTHONPATH`：

```bash
# Linux
export PYTHONPATH=$PYTHONPATH:/usr/local/zed

# macOS
export PYTHONPATH=$PYTHONPATH:/Applications/ZED.app/Contents/Resources/

# Windows
set PYTHONPATH=%PYTHONPATH%;C:\Program Files\Stereolabs\ZED\scripts
```

**验证**：

```bash
python -c "import pyzed.sl; print('ZED OK')"
```

## 步骤 3 — TLS 证书

本地 HTTPS / WSS 需要证书：

```bash
# Linux/macOS
bash ./config/gen_self_signed_cert.sh

# Windows (需要 OpenSSL，或用 Git Bash)
powershell -File ./config/gen_self_signed_cert.ps1
```

**手机访问注意**：Android/iOS 需将 `ca.crt`（根证书）导入系统信任存储。

## 步骤 4 — 启动

```bash
./scripts/run.sh
# 或
python main.py
```

默认监听 `https://0.0.0.0:9000/`。

## 公网访问（手机 VR 盒）

需要 TURN 服务器（STUN 在对称 NAT 下不够用）：

```bash
STEREO_TURN_URLS=turn:your.turn.server:3478 \
STEREO_TURN_USERNAME=user \
STEREO_TURN_PASSWORD=pass \
python main.py
```

## 故障排查

| 问题 | 检查 |
|------|------|
| `pyzed.sl` 找不到 | 确认 ZED SDK 安装 + `PYTHONPATH` 设置正确 |
| 无 ZED 时服务报错 | ZED 懒加载，无 ZED 时服务可启动，其他页不受影响 |
| WebRTC 无法连接 | 确认手机信任了证书；检查 TURN 配置 |
| HTTP 有画面 WebRTC 无 | 问题在 WebRTC signaling 或 ICE，不在相机链路 |

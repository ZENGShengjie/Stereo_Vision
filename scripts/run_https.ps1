# run_https.ps1 — 一键启动 Stereo_Vision（HTTPS 模式）
# 用法: .\scripts\run_https.ps1
# 首次运行前需要先放行脚本执行策略（仅一次）：
#   Set-ExecutionPolicy RemoteSigned -Scope Process

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $PROJECT_ROOT

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 启动服务
python main.py

#!/bin/bash
# 定时任务安装脚本

set -e

echo "=========================================="
echo "TDX Strategy Backtest 定时任务安装"
echo "=========================================="

# 检查是否为 root 用户
if [ "$EUID" -ne 0 ]; then 
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "项目目录: $PROJECT_DIR"

# 方法1: 使用 systemd timer (推荐)
echo ""
echo "方法1: 安装 systemd timer (推荐)"
echo "----------------------------------------"

# 复制 service 和 timer 文件
cp "$SCRIPT_DIR/tdx-backtest-update.service" /etc/systemd/system/
cp "$SCRIPT_DIR/tdx-backtest-update.timer" /etc/systemd/system/

# 重新加载 systemd
systemctl daemon-reload

# 启用并启动 timer
systemctl enable tdx-backtest-update.timer
systemctl start tdx-backtest-update.timer

echo "✅ systemd timer 安装完成"
echo ""
echo "查看 timer 状态:"
echo "  sudo systemctl status tdx-backtest-update.timer"
echo ""
echo "查看下次执行时间:"
echo "  sudo systemctl list-timers tdx-backtest-update.timer"
echo ""
echo "手动触发一次:"
echo "  sudo systemctl start tdx-backtest-update.service"
echo ""
echo "查看执行日志:"
echo "  sudo journalctl -u tdx-backtest-update.service -f"
echo ""

# 方法2: 使用 crontab (备选)
echo "方法2: 使用 crontab (备选)"
echo "----------------------------------------"
echo "如果您更喜欢使用 crontab，请执行:"
echo "  crontab -e"
echo ""
echo "然后添加以下行:"
cat "$SCRIPT_DIR/crontab.txt" | grep "^0 16"
echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="

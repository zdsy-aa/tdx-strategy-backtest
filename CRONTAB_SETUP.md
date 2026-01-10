# 定时任务配置说明

## 问题说明

目前系统**没有配置自动定时任务**，导致数据无法自动下载和更新。需要配置定时任务，让系统每天16:00自动执行数据下载和回测。

## 解决方案

### 方案一：使用 crontab（推荐）

#### 1. 编辑 crontab

```bash
crontab -e
```

#### 2. 添加定时任务

在文件末尾添加以下内容：

```bash
# 每天16:00执行股票数据下载和回测
0 16 * * * cd /home/ubuntu/tdx-strategy-backtest/py_file && /usr/bin/python3 auto_update_daily.py >> /home/ubuntu/tdx-strategy-backtest/logs/cron.log 2>&1
```

**说明**：
- `0 16 * * *`：每天16:00执行
- `cd /home/ubuntu/tdx-strategy-backtest/py_file`：切换到脚本目录
- `/usr/bin/python3 auto_update_daily.py`：执行自动更新脚本
- `>> /home/ubuntu/tdx-strategy-backtest/logs/cron.log 2>&1`：将输出追加到日志文件

#### 3. 验证配置

```bash
# 查看当前的 crontab 任务
crontab -l

# 查看 cron 服务状态
sudo systemctl status cron
```

#### 4. 创建日志目录

```bash
mkdir -p /home/ubuntu/tdx-strategy-backtest/logs
```

#### 5. 手动测试

```bash
cd /home/ubuntu/tdx-strategy-backtest/py_file
python3 auto_update_daily.py
```

---

### 方案二：使用 systemd timer

#### 1. 创建 service 文件

```bash
sudo nano /etc/systemd/system/stock-backtest.service
```

内容：

```ini
[Unit]
Description=Stock Data Download and Backtest
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/tdx-strategy-backtest/py_file
ExecStart=/usr/bin/python3 /home/ubuntu/tdx-strategy-backtest/py_file/auto_update_daily.py
StandardOutput=append:/home/ubuntu/tdx-strategy-backtest/logs/systemd.log
StandardError=append:/home/ubuntu/tdx-strategy-backtest/logs/systemd.log

[Install]
WantedBy=multi-user.target
```

#### 2. 创建 timer 文件

```bash
sudo nano /etc/systemd/system/stock-backtest.timer
```

内容：

```ini
[Unit]
Description=Stock Data Download and Backtest Timer
Requires=stock-backtest.service

[Timer]
OnCalendar=*-*-* 16:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

#### 3. 启用并启动 timer

```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启用 timer（开机自启）
sudo systemctl enable stock-backtest.timer

# 启动 timer
sudo systemctl start stock-backtest.timer

# 查看 timer 状态
sudo systemctl status stock-backtest.timer

# 查看所有 timer
systemctl list-timers --all
```

#### 4. 手动触发任务（测试）

```bash
sudo systemctl start stock-backtest.service
```

---

## 日志查看

### crontab 日志

```bash
# 查看 cron 日志
tail -f /home/ubuntu/tdx-strategy-backtest/logs/cron.log

# 查看自动更新脚本生成的日志
ls -lht /home/ubuntu/tdx-strategy-backtest/logs/auto_update_*.log
tail -f /home/ubuntu/tdx-strategy-backtest/logs/auto_update_$(date +%Y%m%d).log
```

### systemd 日志

```bash
# 查看 systemd 日志
tail -f /home/ubuntu/tdx-strategy-backtest/logs/systemd.log

# 查看 systemd journal 日志
sudo journalctl -u stock-backtest.service -f
```

---

## 常见问题

### 1. 任务没有执行

**检查 cron 服务**：
```bash
sudo systemctl status cron
sudo systemctl restart cron
```

**检查 crontab 语法**：
```bash
crontab -l
```

### 2. Python 路径错误

**查找 Python 路径**：
```bash
which python3
```

**更新 crontab 中的 Python 路径**。

### 3. 权限问题

**确保脚本有执行权限**：
```bash
chmod +x /home/ubuntu/tdx-strategy-backtest/py_file/auto_update_daily.py
```

**确保日志目录可写**：
```bash
mkdir -p /home/ubuntu/tdx-strategy-backtest/logs
chmod 755 /home/ubuntu/tdx-strategy-backtest/logs
```

### 4. 时区问题

**检查系统时区**：
```bash
timedatectl
```

**设置时区为中国（如果需要）**：
```bash
sudo timedatectl set-timezone Asia/Shanghai
```

---

## 验证任务执行

### 方法1：查看数据文件修改时间

```bash
ls -lt /home/ubuntu/tdx-strategy-backtest/data/day/*/*.csv | head -20
```

### 方法2：查看日志文件

```bash
ls -lht /home/ubuntu/tdx-strategy-backtest/logs/
```

### 方法3：查看 JSON 数据更新时间

```bash
ls -l /home/ubuntu/tdx-strategy-backtest/web/client/src/data/*.json
```

---

## 推荐配置

**生产环境推荐使用 crontab**，因为：
1. 配置简单
2. 日志清晰
3. 易于调试
4. 广泛使用

**配置命令**：

```bash
# 1. 创建日志目录
mkdir -p /home/ubuntu/tdx-strategy-backtest/logs

# 2. 编辑 crontab
crontab -e

# 3. 添加以下内容
0 16 * * * cd /home/ubuntu/tdx-strategy-backtest/py_file && /usr/bin/python3 auto_update_daily.py >> /home/ubuntu/tdx-strategy-backtest/logs/cron.log 2>&1

# 4. 保存并退出

# 5. 验证配置
crontab -l
```

---

## 当前状态

- ❌ **未配置定时任务**
- ✅ 自动更新脚本已就绪（`py_file/auto_update_daily.py`）
- ✅ 数据下载脚本已就绪（`py_file/stock_downloader_script.py`）
- ✅ 回测脚本已就绪（`py_file/full_backtest.py`）

**下一步**：按照上述说明配置 crontab 定时任务。

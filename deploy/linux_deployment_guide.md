# TDX 策略回测系统 - Linux 云服务器部署方案

**版本**: 1.0.0
**日期**: 2026-01-09
**作者**: Manus AI

---

## 1. 概述

本方案旨在指导用户如何在标准的 Linux 云服务器（如阿里云 ECS, 腾讯云 CVM, AWS EC2 等）上完整部署 `tdx-strategy-backtest` 项目。部署包含三个核心部分：

1.  **后端环境**: Python 脚本，用于数据下载和回测计算。
2.  **前端应用**: React 静态网站，用于数据可视化展示。
3.  **自动化任务**: 定时执行数据更新和回测的 systemd timer 或 crontab。

---

## 2. 服务器要求

### 2.1. 硬件配置 (最低要求)

- **操作系统**: Ubuntu 22.04 LTS (推荐) 或 CentOS 7+
- **CPU**: 2 核
- **内存**: 4 GB
- **磁盘**: 20 GB (主要用于存储股票历史数据)

### 2.2. 软件环境

- **Git**: 用于克隆代码仓库。
- **Python**: 3.10 或更高版本。
- **Node.js**: 18.x 或更高版本。
- **pnpm**: 用于高效管理 Node.js 依赖。
- **Nginx**: (推荐) 作为 Web 服务器，托管前端静态文件并提供反向代理。

---

## 3. 部署步骤

以下步骤以 **Ubuntu 22.04** 为例。假设您已通过 SSH 登录到您的云服务器。

### 步骤 1: 安装基础环境

```bash
# 更新系统包列表
sudo apt update && sudo apt upgrade -y

# 安装 Git, Python, pip, Nginx
sudo apt install -y git python3 python3-pip python3-venv nginx

# 安装 Node.js (使用 nvm 管理版本)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 使 nvm 生效
source ~/.bashrc

# 安装 Node.js 18
nvm install 18
nvm use 18

# 安装 pnpm
corepack enable
```

### 步骤 2: 克隆项目并设置后端

```bash
# 克隆项目到用户主目录
cd ~
gh repo clone zdsy-aa/tdx-strategy-backtest
cd tdx-strategy-backtest

# 创建并激活 Python 虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 退出虚拟环境 (后续脚本会自动调用)
deactivate
```

### 步骤 3: 首次数据下载与回测

这一步是为了生成初始的前端数据。**此过程耗时较长，建议在 `screen` 或 `tmux` 会话中执行，以防 SSH 连接中断。**

```bash
# 进入 Python 脚本目录
cd ~/tdx-strategy-backtest/py_file

# 激活虚拟环境
source ../venv/bin/activate

# 1. 下载股票数据 (首次运行约需30-60分钟)
python3 stock_downloader_script.py

# 2. 执行完整回测 (计算量大，可能需要数小时)
python3 full_backtest.py

# 退出虚拟环境
deactivate
```

### 步骤 4: 构建并部署前端

```bash
# 进入前端项目目录
cd ~/tdx-strategy-backtest/web

# 安装前端依赖
pnpm install

# 构建生产版本的静态文件
pnpm build
```

构建完成后，所有静态文件将生成在 `web/dist` 目录下。

### 步骤 5: 配置 Nginx

创建一个新的 Nginx 配置文件来托管前端应用。

```bash
# 创建 Nginx 配置文件
sudo nano /etc/nginx/sites-available/tdx-backtest
```

将以下内容粘贴到文件中，并保存退出 (`Ctrl+X`, `Y`, `Enter`)。

```nginx
server {
    listen 80;
    server_name your_domain.com; # 替换为您的域名或服务器IP地址

    # 前端静态文件根目录
    root /home/ubuntu/tdx-strategy-backtest/web/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # 日志文件
    access_log /var/log/nginx/tdx-backtest.access.log;
    error_log /var/log/nginx/tdx-backtest.error.log;
}
```

**启用 Nginx 配置**: 

```bash
# 创建软链接以启用站点
sudo ln -s /etc/nginx/sites-available/tdx-backtest /etc/nginx/sites-enabled/

# 测试 Nginx 配置是否有语法错误
sudo nginx -t

# 重启 Nginx 服务使配置生效
sudo systemctl restart nginx
```

现在，您应该可以通过服务器的 IP 地址或域名在浏览器中访问回测系统了。

### 步骤 6: 安装自动化定时任务

为了让系统每天自动更新数据，我们需要安装之前创建的定时任务。

```bash
# 进入部署脚本目录
cd ~/tdx-strategy-backtest/deploy

# 运行安装脚本 (需要 sudo 权限)
sudo ./install_timer.sh
```

该脚本会自动安装并启动 `systemd timer`。您可以运行以下命令来验证安装：

- **查看 timer 状态**: `sudo systemctl status tdx-backtest-update.timer`
- **查看下次执行时间**: `sudo systemctl list-timers tdx-backtest-update.timer`

---

## 4. 系统维护与更新

### 4.1. 更新代码

当项目代码有更新时，可以执行以下步骤进行更新：

```bash
# 进入项目目录
cd ~/tdx-strategy-backtest

# 拉取最新代码
git pull origin main

# 如果后端有依赖更新
source venv/bin/activate
pip install -r requirements.txt
deactivate

# 如果前端有更新，重新构建
cd web
pnpm install
pnpm build

# 重启 Nginx (如果配置有变动)
sudo systemctl restart nginx
```

### 4.2. 查看日志

- **自动化任务日志**: `sudo journalctl -u tdx-backtest-update.service -f`
- **Nginx 访问日志**: `tail -f /var/log/nginx/tdx-backtest.access.log`

---

## 5. 常见问题 (FAQ)

- **首次回测时间过长怎么办？**
  - 这是正常现象，因为需要处理数千只股票多年的数据。请务必在 `screen` 或 `tmux` 中运行，耐心等待其完成。

- **Nginx 访问出现 403 Forbidden 错误？**
  - 检查 `/home/ubuntu/tdx-strategy-backtest` 及其子目录的文件权限，确保 Nginx 用户 (`www-data`) 有读取权限。

- **定时任务没有按时执行？**
  - 使用 `sudo systemctl status tdx-backtest-update.timer` 和 `sudo journalctl -u tdx-backtest-update.service` 检查 timer 和 service 的状态及日志，排查错误。

@echo off
echo ==========================================
echo 通达信指标回测系统 - Windows 依赖安装脚本
echo ==========================================

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8 或更高版本。
    echo 请访问 https://www.python.org/downloads/ 下载并安装。
    echo 安装时请务必勾选 "Add Python to PATH"。
    pause
    exit /b
)

:: 检查 Git 是否安装
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到 Git，建议安装 Git 以便管理代码。
    echo 请访问 https://git-scm.com/download/win 下载安装。
)

echo [1/3] 正在创建虚拟环境 (venv)...
python -m venv venv

echo [2/3] 正在激活虚拟环境...
call .\venv\Scripts\activate

echo [3/3] 正在安装 Python 依赖项 (不包含 Web 相关)...
pip install --upgrade pip
pip install pandas>=1.5.0 numpy>=1.21.0 akshare>=1.10.0 requests>=2.28.0 matplotlib>=3.5.0 seaborn>=0.12.0 tqdm>=4.64.0

echo ==========================================
echo 依赖安装完成！
echo 您现在可以运行回测脚本了。
echo ==========================================
pause

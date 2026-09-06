@echo off
chcp 65001 >nul
title 垂直轴风轮载荷数据处理系统 - 启动器
echo ================================================
echo  垂直轴风轮载荷数据处理系统 - 启动器
echo ================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+ 
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
python -c "import numpy, pandas, matplotlib, openpyxl, reportlab, docx, PIL" >nul 2>&1
if errorlevel 1 (
    echo [提示] 检测到缺少依赖，正在安装...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络后重试
        pause
        exit /b 1
    )
    echo [完成] 依赖安装成功
) else (
    echo [完成] 依赖已就绪
)

echo [2/3] 启动 Web 服务...
echo  服务地址: http://127.0.0.1:8080
echo  启动后请用浏览器打开上述地址
echo.

REM 限制 BLAS/OpenMP 线程，避免 NumPy 启动时一次性申请过多线程内存
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"

REM 启动服务
python web_gui.py

pause

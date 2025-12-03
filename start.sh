#!/bin/bash
# 启动 web_chat_app
echo "=== Web Chat App 启动脚本 ==="

# 创建必要目录
mkdir -p uploads downloads conversations

# 检查Python依赖
echo "检查依赖..."
pip install -r requirements.txt -q

# 启动应用
echo "启动Web服务器..."
echo "请访问: http://localhost:5000"
python web_chat_app.py

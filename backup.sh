#!/bin/bash
cd "$(dirname "$0")"
git add .
git commit -m "backup $(date)" 2>/dev/null
git push origin main

##!/bin/bash
#
## === 配置区（按你的情况改）===
#REPO_NAME="NeuralNetwork"
#GITHUB_USER="shibo45678"
## ============================
#
#echo "🚀 开始自动备份到 GitHub..."
#
## 进入脚本所在目录（确保在项目根目录运行）
#cd "$(dirname "$0")"
#
## 1. 添加所有被 Git 跟踪的文件的改动（忽略 .gitignore 的内容）
#git add .
#
## 2. 检查是否有实际改动
#if ! git diff --cached --quiet; then
#    echo "📝 有改动，正在提交..."
#    # 自动生成带时间的提交信息
#    git commit -m "Auto backup: $(date '+%Y-%m-%d %H:%M')"
#
#    # 3. 推送到 GitHub
#    echo "📤 正在推送到 GitHub..."
#    git push origin main 2>&1 | grep -v "Everything up-to-date"
#
#    if [ $? -eq 0 ]; then
#        echo "✅ 备份成功！"
#    else
#        echo "❌ 推送失败，请检查网络或权限。"
#        exit 1
#    fi
#else
#    echo "💤 没有新改动，无需备份。"
#fi
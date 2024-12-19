#!/bin/bash
echo "注意：以 python3.10.14 为基础环境" 
echo "1.如果需要模型下载请从魔搭下载,参考download.py下载,python download.py,下载到当前目录"
echo "2. 安装python3.10.14版本的环境依赖包(尽可能安装依赖的包，排除无关包，避免安装过慢,指定国内源)，pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && python download.py
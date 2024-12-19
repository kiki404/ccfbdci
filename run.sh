#!/bin/bash
TEST_DATASETS_DIR=$1
echo "1. 传入数据集目录 $1"
echo "2. 固定输出目录 /output/answer.jsonl"
echo "运行代码"
python ./your_project/test.py ${TEST_DATASETS_DIR}
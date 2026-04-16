#!/bin/bash

# 检查是否输入了显卡ID号
if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

# 获取显卡ID号
gpu_id=$1
device="/dev/nvidia${gpu_id}"

# 获取所有使用该设备的进程ID
pids=$(fuser -v $device 2>/dev/null | grep -oP '\d+')

# 检查是否有进程
if [ -z "$pids" ]; then
    echo "No processes found using $device."
    exit 0
fi

# 显示将要删除的进程
echo "The following processes will be terminated:"
echo $pids

# 终止这些进程
for pid in $pids; do
    echo "Terminating process $pid"
    kill -9 $pid
done
pkill -9 -f spawn_main
echo "All processes terminated."

#!/bin/bash

# 修复版僵尸进程清理工具
echo "=== 僵尸进程清理工具 ==="

# 检查权限
if [ "$EUID" -ne 0 ]; then
    echo "建议使用 root 权限运行此脚本以获得最佳效果"
fi

# 统计初始僵尸进程数量
initial_count=$(ps aux | awk '$8=="Z" {count++} END {print count+0}')
echo "初始僵尸进程数量: ${initial_count:-0}"

if [ "${initial_count:-0}" -eq 0 ]; then
    echo "系统很干净，没有发现僵尸进程。"
    exit 0
fi

echo ""
echo "发现的僵尸进程详情："
ps -eo pid,ppid,state,comm | awk '$3=="Z" {print "PID:", $1, "PPID:", $2, "CMD:", $4}'

echo ""
echo "开始清理..."

# 清理僵尸进程
cleaned=0
ps -eo pid,ppid,state | awk '$3=="Z" {print $1, $2}' | while read zombie_pid parent_pid; do
    if [ -n "$zombie_pid" ] && [ -n "$parent_pid" ]; then
        if [ "$parent_pid" -ne 1 ] 2>/dev/null; then
            echo "尝试清理: 僵尸进程 $zombie_pid (父进程: $parent_pid)"
            
            # 先尝试发送 SIGCHLD
            kill -s SIGCHLD "$parent_pid" 2>/dev/null
            sleep 0.5
            
            # 检查是否还存在，如果存在则杀死父进程
            if ps -p "$zombie_pid" >/dev/null 2>&1; then
                echo "正常方式失败，尝试杀死父进程 $parent_pid"
                kill -9 "$parent_pid" 2>/dev/null
                cleaned=$((cleaned + 1))
            else
                cleaned=$((cleaned + 1))
            fi
        else
            echo "跳过: 僵尸进程 $zombie_pid (父进程是 init，需要重启相关服务)"
        fi
    fi
done

# 等待一下让系统处理
sleep 2

# 检查清理结果
final_count=$(ps aux | awk '$8=="Z" {count++} END {print count+0}')
echo ""
echo "清理完成！"
echo "剩余僵尸进程数量: ${final_count:-0}"

if [ "${final_count:-0}" -gt 0 ]; then
    echo "以下僵尸进程仍然存在（可能需要重启系统或相关服务）："
    ps -eo pid,ppid,state,comm | awk '$3=="Z" {print "PID:", $1, "PPID:", $2, "CMD:", $4}'
fi
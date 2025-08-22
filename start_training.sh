#!/bin/bash
# 启动训练脚本 - 支持继续训练和最佳模型保存

echo "=== RadioML 训练启动脚本 ==="
echo "时间: $(date)"
echo

# 显示帮助信息
show_help() {
    echo "使用方法:"
    echo "  ./start_training.sh                    # 开始新训练 (250 epochs)"
    echo "  ./start_training.sh --continue         # 从检查点继续训练"
    echo "  ./start_training.sh --epochs 100      # 指定训练轮数"
    echo "  ./start_training.sh --continue --epochs 300  # 继续训练并指定轮数"
    echo ""
    echo "选项:"
    echo "  --continue      从最新检查点继续训练"
    echo "  --epochs N      设置训练轮数 (默认: 250)"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "新功能:"
    echo "  ✓ 自动保存最佳模型到 TensorBoard 文件夹"
    echo "  ✓ 每5轮自动保存检查点"
    echo "  ✓ 支持断点续训"
    echo "  ✓ TensorBoard 记录最佳准确度"
    exit 0
}

# 解析命令行参数
CONTINUE_TRAIN=""
EPOCHS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_TRAIN="--continue_train"
            shift
            ;;
        --epochs)
            EPOCHS="--epochs $2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo

# 检查现有检查点
if [ -n "$CONTINUE_TRAIN" ]; then
    echo "=== 检查点状态 ==="
    checkpoint_files=(runs/checkpoint_*.pth)
    if [ -e "${checkpoint_files[0]}" ]; then
        latest_checkpoint=$(ls -t runs/checkpoint_*.pth | head -n 1)
        echo "找到检查点: $latest_checkpoint"
        # 提取epoch信息
        epoch_num=$(basename "$latest_checkpoint" | sed 's/checkpoint_\([0-9]*\)\.pth/\1/')
        echo "将从第 $((epoch_num + 1)) 轮继续训练"
    else
        echo "⚠️  未找到检查点文件，将从头开始训练"
        CONTINUE_TRAIN=""
    fi
    echo
fi

# 检查配置
echo "=== 当前配置 ==="
echo "训练模式: $([ -n "$CONTINUE_TRAIN" ] && echo "继续训练" || echo "新训练")"
echo "训练轮数: $([ -n "$EPOCHS" ] && echo "${EPOCHS#--epochs }" || echo "250")"
echo "GPU优化: 已启用"
echo "新功能: 最佳模型保存 + 检查点 + TensorBoard"
echo

# 设置环境变量以优化性能
export PYTHONPATH="/home/jiangrundong/hy_bak_test_delete_after_used/complex_gru_single_radioml:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=0  # 启用异步CUDA调用
export TORCH_COMPILE_DEBUG=0   # 禁用torch.compile调试信息

# 检查是否有现有的训练进程
existing_processes=$(ps aux | grep "python.*main.py" | grep -v grep | wc -l)
if [ $existing_processes -gt 0 ]; then
    echo "⚠️  检测到现有的训练进程:"
    ps aux | grep "python.*main.py" | grep -v grep
    echo
    read -p "是否继续启动新的训练? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "取消启动"
        exit 0
    fi
fi

# 构建训练命令
CMD="python main.py $CONTINUE_TRAIN $EPOCHS"

echo "=== 启动训练 ==="
echo "执行命令: $CMD"
echo "提示: 使用Ctrl+C停止训练"
echo "监控命令: tensorboard --logdir=runs"
echo "GPU监控: watch -n 1 nvidia-smi"
echo

# 启动训练
$CMD

echo
echo "=== 训练结束 ==="
echo "时间: $(date)"

# 显示训练结果
echo
echo "=== 训练结果 ==="
if [ -d "runs" ]; then
    echo "TensorBoard 日志目录:"
    ls -la runs/ | grep "^d" | tail -5
    
    echo
    echo "最佳模型位置:"
    find runs/ -name "best_model.pth" -exec ls -la {} \; 2>/dev/null | tail -3
    
    echo
    echo "检查点文件:"
    ls -la runs/checkpoint_*.pth 2>/dev/null | tail -5
fi

# 显示最终GPU状态
echo
echo "=== 最终GPU状态 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo
echo "=== 使用TensorBoard查看训练过程 ==="
echo "命令: tensorboard --logdir=runs"
echo "然后在浏览器中打开 http://localhost:6006"

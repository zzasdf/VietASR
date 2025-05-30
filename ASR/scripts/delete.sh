#!/bin/bash
# 删除指定范围内的 epoch 文件
# 用法: ./delete_epochs.sh <文件夹路径> <起始编号> <结束编号>

if [ $# -ne 3 ]; then
    echo "错误：参数数量不正确"
    echo "用法: $0 <文件夹> <起始epoch> <结束epoch>"
    exit 1
fi

DIR="$1"
START="$2"
END="$3"

# 验证参数有效性
if [ ! -d "$DIR" ]; then
    echo "错误：目录 '$DIR' 不存在"
    exit 1
fi

if ! [[ "$START" =~ ^[0-9]+$ ]] || ! [[ "$END" =~ ^[0-9]+$ ]]; then
    echo "错误：起始和结束值必须是整数"
    exit 1
fi

if [ "$START" -gt "$END" ]; then
    echo "错误：起始值不能大于结束值"
    exit 1
fi

# 构建文件匹配模式[2,7](@ref)
PATTERN=""
for ((i=START; i<=END; i++)); do
    PATTERN="$PATTERN epoch-$i.pt"
done

# 安全删除文件[1,3](@ref)
echo "即将删除以下文件："
find "$DIR" -maxdepth 1 -type f -name "epoch-*.pt" | sort -V | 
    awk -v start="$START" -v end="$END" '
    {
    filename = $0
    gsub(/.*\//, "", filename)
    if (match(filename, /epoch-([0-9]+)\.pt/, arr)) {
        epoch = arr[1]
        if (epoch >= start && epoch <= end) {
        print $0
        file_count++
        }
    }
    } END {
        if (file_count == 0) print "未找到匹配文件";
    }'

read -p "确认删除? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "操作已取消"
    exit 0
fi

# 执行删除[1,7](@ref)
find "$DIR" -maxdepth 1 -type f -name "epoch-*.pt" -print0 | 
    while IFS= read -r -d $'\0' file; do
        epoch=$(echo "$file" | grep -oP 'epoch-\K\d+(?=\.pt)')
        if [ "$epoch" -ge "$START" ] && [ "$epoch" -le "$END" ]; then
            rm -v "$file"
        fi
    done

echo "删除操作完成"

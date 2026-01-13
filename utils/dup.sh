cd ./data

for part in test  ; do
    trans_file="$part/tongdai_clean/test.trans.txt"
    if [ -f "$trans_file" ]; then
        echo "Processing $trans_file"
        
        # Backup
        cp "$trans_file" "$trans_file.backup"
        
        # Loại bỏ duplicate (giữ dòng đầu tiên)
        awk '!seen[$1]++' "$trans_file.backup" > "$trans_file"
        
        # Kiểm tra
        total=$(wc -l < "$trans_file.backup")
        after=$(wc -l < "$trans_file")
        removed=$((total - after))
        echo "  Total: $total, After: $after, Removed: $removed"
    fi
done
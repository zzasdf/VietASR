cd /vietasr/data1 

# Fix tất cả file .trans.txt
find . -name "*.trans.txt" | while read file; do
    echo "Fixing: $file"
    # Backup
    cp "$file" "$file.bak"
    # Chỉ giữ dòng có text (>= 2 fields)
    awk 'NF>=2' "$file.bak" > "$file"
    # Count dòng bị xóa
    removed=$(($(wc -l < "$file.bak") - $(wc -l < "$file")))
    echo "  Removed $removed invalid lines"
done
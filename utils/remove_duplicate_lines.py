#!/usr/bin/env python3
"""
Script để xóa các dòng trùng lặp từ transcripts.txt
Giữ lại dòng đầu tiên, xóa các dòng trùng lặp tiếp theo
"""
import sys
from pathlib import Path

def remove_duplicate_lines(transcripts_file, duplicate_lines_file):
    """Xóa các dòng được chỉ định trong duplicate_lines.txt"""
    
    # Đọc danh sách line numbers cần xóa
    with open(duplicate_lines_file, 'r') as f:
        lines_to_remove = set(int(line.strip()) for line in f if line.strip())
    
    print(f"Đang xóa {len(lines_to_remove)} dòng trùng lặp...")
    
    # Tạo backup
    backup_file = Path(transcripts_file).with_suffix('.txt.bak')
    Path(transcripts_file).rename(backup_file)
    print(f"Đã tạo backup: {backup_file}")
    
    # Đọc và ghi lại file, bỏ qua các dòng trùng
    kept_count = 0
    removed_count = 0
    
    with open(backup_file, 'r', encoding='utf-8') as infile:
        with open(transcripts_file, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                if line_num in lines_to_remove:
                    removed_count += 1
                else:
                    outfile.write(line)
                    kept_count += 1
    
    print(f"✓ Hoàn tất!")
    print(f"  - Số dòng giữ lại: {kept_count}")
    print(f"  - Số dòng đã xóa: {removed_count}")
    print(f"  - File gốc đã backup: {backup_file}")
    print(f"  - File mới: {transcripts_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_duplicate_lines.py <transcripts_file> <duplicate_lines_file>")
        sys.exit(1)
    
    transcripts_file = sys.argv[1]
    duplicate_lines_file = sys.argv[2]
    
    if not Path(transcripts_file).exists():
        print(f"Error: File không tồn tại: {transcripts_file}")
        sys.exit(1)
    
    if not Path(duplicate_lines_file).exists():
        print(f"Error: File không tồn tại: {duplicate_lines_file}")
        sys.exit(1)
    
    remove_duplicate_lines(transcripts_file, duplicate_lines_file)

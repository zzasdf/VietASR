#!/usr/bin/env python3
"""
Script để tìm các ID trùng lặp trong file transcripts.txt
"""
import sys
from pathlib import Path
from collections import Counter

def find_duplicates(transcripts_file):
    """Tìm các filename trùng lặp trong transcripts.txt"""
    seen_ids = []
    
    with open(transcripts_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            
            rel_wav_path = parts[0].strip()
            filename = Path(rel_wav_path).name
            recording_id = Path(filename).stem  # Lấy filename không có extension
            
            seen_ids.append((recording_id, line_num, line.strip()))
    
    # Đếm số lần xuất hiện của mỗi ID
    id_counter = Counter([item[0] for item in seen_ids])
    
    # Tìm các ID xuất hiện nhiều hơn 1 lần
    duplicates = {id_: count for id_, count in id_counter.items() if count > 1}
    
    if not duplicates:
        print("✓ Không tìm thấy ID trùng lặp!")
        return
    
    print(f"✗ Tìm thấy {len(duplicates)} ID bị trùng lặp:")
    print(f"  Tổng số dòng trùng: {sum(duplicates.values())}\n")
    
    # In chi tiết các ID trùng
    for dup_id in sorted(duplicates.keys()):
        count = duplicates[dup_id]
        print(f"\n{'='*80}")
        print(f"ID: {dup_id} (xuất hiện {count} lần)")
        print(f"{'='*80}")
        
        # Tìm tất cả các dòng chứa ID này
        dup_lines = [item for item in seen_ids if item[0] == dup_id]
        for recording_id, line_num, line_content in dup_lines:
            print(f"  Dòng {line_num}: {line_content[:100]}...")
    
    # Export danh sách line numbers để xóa (giữ lại dòng đầu tiên, xóa các dòng sau)
    lines_to_remove = []
    for dup_id in duplicates.keys():
        dup_lines = [item for item in seen_ids if item[0] == dup_id]
        # Giữ dòng đầu tiên, xóa các dòng còn lại
        for _, line_num, _ in dup_lines[1:]:
            lines_to_remove.append(line_num)
    
    print(f"\n{'='*80}")
    print(f"GIẢI PHÁP:")
    print(f"{'='*80}")
    print(f"Tổng số dòng cần xóa: {len(lines_to_remove)}")
    
    # Lưu danh sách line numbers vào file
    output_file = Path(transcripts_file).parent / "duplicate_lines.txt"
    with open(output_file, 'w') as f:
        for line_num in sorted(lines_to_remove):
            f.write(f"{line_num}\n")
    
    print(f"\nĐã lưu danh sách line numbers cần xóa vào: {output_file}")
    print(f"\nSử dụng script remove_duplicate_lines.py để tự động xóa các dòng trùng lặp.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_duplicates.py <transcripts_file>")
        sys.exit(1)
    
    transcripts_file = sys.argv[1]
    if not Path(transcripts_file).exists():
        print(f"Error: File không tồn tại: {transcripts_file}")
        sys.exit(1)
    
    find_duplicates(transcripts_file)

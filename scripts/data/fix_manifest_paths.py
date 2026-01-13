#!/usr/bin/env python3
"""
Fix paths in manifest files to use absolute paths compatible with Docker.
"""
import gzip
import json
import sys
from pathlib import Path

def fix_paths_in_manifest(input_file, output_file=None):
    """Fix relative paths to absolute paths in manifest."""
    if output_file is None:
        output_file = input_file
    
    # Backup original if modifying in place
    if input_file == output_file:
        backup = f"{input_file}.backup"
        Path(input_file).rename(backup)
        input_file = backup
    
    count = 0
    fixed = 0
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
        
        for line in f_in:
            count += 1
            data = json.loads(line)
            
            # Fix features storage_path
            if 'features' in data and 'storage_path' in data['features']:
                old_path = data['features']['storage_path']
                if old_path.startswith('../'):
                    # Convert ../data3/fbank -> /vietasr/data3/fbank
                    new_path = old_path.replace('../', '/vietasr/')
                    data['features']['storage_path'] = new_path
                    fixed += 1
            
            # Fix recording source paths
            if 'recording' in data and 'sources' in data['recording']:
                for source in data['recording']['sources']:
                    if 'source' in source:
                        old_source = source['source']
                        if old_source.startswith('../'):
                            new_source = old_source.replace('../', '/vietasr/')
                            source['source'] = new_source
                            fixed += 1
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Processed {count} lines, fixed {fixed} paths")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_manifest_paths.py <manifest_file.jsonl.gz>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    fix_paths_in_manifest(input_file)

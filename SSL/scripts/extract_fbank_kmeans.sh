#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
python scripts/kmeans/extract_kmeans.py run --task-list ${task_file} --model-path ${kmeans_model_path}
import soundfile as sf
import os
import torch
import math
import fcntl
from tqdm import tqdm
from argparse import ArgumentParser
from funasr import AutoModel

def routine(wav_file, tgt_dir, model, device, args):
    speech, sample_rate = sf.read(wav_file)
    os.makedirs(tgt_dir, exist_ok=True)
    if args.streaming:
        chunk_size = 200 # ms
        chunk_stride = int(chunk_size * sample_rate / 1000)

        cache = {}
        total_chunk_num = int(len((speech)-1)/chunk_stride+1)
        res = [{
            "key":os.path.basename(wav_file),
            "value" : []
        }]
        for i in range(total_chunk_num):
            speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
            is_final = i == total_chunk_num - 1
            tem_res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, disable_pbar=True, device=device)
            if len(tem_res[0]["value"]):
                if tem_res[0]["value"][0][0]==-1:
                    res[0]["value"][-1][1] = tem_res[0]["value"][0][1]
                else:
                    res[0]["value"].append(tem_res[0]["value"][0])

                for item in tem_res[0]["value"][1:]:
                    res[0]["value"].append(item)

    else:
        res = model.generate(input=wav_file, disable_pbar=True, device=device)


    if args.max_duration is not None:
        max_size = int(args.max_duration*sample_rate)
    for i, item in enumerate(res[0]["value"]):
        start_index = int(item[0]*sample_rate/1000)
        end_index = int(item[1]*sample_rate/1000)
        segments = []
        if args.max_duration is not None and end_index - start_index  > max_size:
            segment_n = round((end_index-start_index)/max_size)
            step = math.ceil((end_index-start_index)/segment_n)
            for lindex in range(start_index, end_index, step):
                rindex = min(lindex+step, end_index)
                if rindex<=lindex:
                    continue
                segments.append(speech[lindex: rindex])
        else:
            segments = [speech[start_index: end_index]]
        for j, segment in enumerate(segments):
            sf.write(os.path.join(tgt_dir, f"{i}-{j}.wav"), segment, sample_rate)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-duration", type = float, default=None)
    parser.add_argument("--task-dir", type=str)
    parser.add_argument("--task-block", type=int,default=1)
    args = parser.parse_args()
    # os.makedirs(args.tgt_dir, exist_ok=True)

    task_dir = args.task_dir
    with open(os.path.join(task_dir, "running_task"), 'r') as f_task:
        task_lines = f_task.read().splitlines()

    task_block = args.task_block

    device = torch.device("cuda:0")
    model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", device="cuda:0", max_end_silence_time=500)
    # model.to(device)
    while True:
        with open(os.path.join(task_dir, "lock"), 'w') as f_lock:
            fcntl.flock(f_lock.fileno(), fcntl.LOCK_EX)
            with open(os.path.join(task_dir, "index"), 'r') as f_index:
                task_index = f_index.readlines()
                task_index = int(task_index[0])
            with open(os.path.join(task_dir, "index"), 'w') as f_index:
                f_index.write(f"{task_index+task_block}")

        if task_index >= len(task_lines):
            break
        tasks = task_lines[task_index: task_index+task_block]
        done_tasks = []
        for task in tqdm(tasks):
            task_split = task.split()
            wav_file = task_split[0]
            save_dir = task_split[1]
            # convert video to wav
            try:
                routine(wav_file, save_dir, model, device, args)
                # process subtitle and video info
            except KeyboardInterrupt:
                raise
            except Exception:
                continue
            done_tasks.append(task)

        with open(os.path.join(task_dir, "lock"), 'w') as f_lock:
            fcntl.flock(f_lock.fileno(), fcntl.LOCK_EX)
            with open(os.path.join(task_dir, "done"), 'a') as f_done:
                for task in done_tasks:
                    print(task, file=f_done)


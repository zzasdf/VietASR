import fcntl
import math
import multiprocessing as mp
import os
from argparse import ArgumentParser

import soundfile as sf
import torch
from tqdm import tqdm


def routine(wav_file, tgt_dir, model, device, args):
    try:
        speech, sample_rate = sf.read(wav_file)
    except KeyboardInterrupt:
        raise
    except:
        return []

    if speech.shape[0] / sample_rate <= args.max_duration:
        if speech.shape[0] / sample_rate <= args.min_duration:
            return []
        if tgt_dir is not None:
             os.system(f"cp {wav_file} {tgt_dir}.wav")
        # Return the whole file as one segment
        return [[0, speech.shape[0] / sample_rate]]

    if tgt_dir is not None:
        os.makedirs(tgt_dir, exist_ok=True)

    if args.streaming:
        chunk_size = 200  # ms
        chunk_stride = int(chunk_size * sample_rate / 1000)

        cache = {}
        total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
        res = [{"key": os.path.basename(wav_file), "value": []}]
        for i in range(total_chunk_num):
            speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            tem_res = model.generate(
                input=speech_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                disable_pbar=True,
                device=device,
            )
            if len(tem_res[0]["value"]):
                if tem_res[0]["value"][0][0] == -1:
                    res[0]["value"][-1][1] = tem_res[0]["value"][0][1]
                else:
                    res[0]["value"].append(tem_res[0]["value"][0])

                for item in tem_res[0]["value"][1:]:
                    res[0]["value"].append(item)

    else:
        res = model.generate(input=wav_file, disable_pbar=True, device=device)

    def merge_segment(segment, merge_time, min_time, max_blank_time):
        new_value = []
        start = None
        end = None

        if merge_time is None:
            return segment

        for item in segment:
            start_time = item[0] / 1000
            end_time = item[1] / 1000
            if start is None:
                start = item[0]
                end = item[1]
            else:
                if end / 1000 - start / 1000 > merge_time:
                    new_value.append([start, end])
                    start = item[0]
                    end = item[1]
                elif start_time - end / 1000 > max_blank_time:
                    # not merget when there is too much blank
                    if end / 1000 - start / 1000 > min_time:
                        new_value.append([start, end])
                    start = item[0]
                    end = item[1]
                else:
                    end = item[1]

        if start is not None and end / 1000 - start / 1000 > min_time:
            new_value.append([start, end])

        return new_value

    res[0]["value"] = merge_segment(
        res[0]["value"], args.merge_duration, args.min_duration, args.max_blank_duration
    )

    if args.max_duration is not None:
        max_size = int(args.max_duration * sample_rate)
    
    final_segments = []
    for i, item in enumerate(res[0]["value"]):
        start_index = int(item[0] * sample_rate / 1000)
        end_index = int(item[1] * sample_rate / 1000)
        segments = []
        if args.max_duration is not None and end_index - start_index > max_size:
            segment_n = round((end_index - start_index) / max_size)
            step = math.ceil((end_index - start_index) / segment_n)
            for lindex in range(start_index, end_index, step):
                rindex = min(lindex + step, end_index)
                if rindex <= lindex:
                    continue
                segments.append((speech[lindex:rindex], lindex, rindex))
        else:
            segments = [(speech[start_index:end_index], start_index, end_index)]
        
        for j, (segment, seg_start, seg_end) in enumerate(segments):
            if tgt_dir is not None:
                sf.write(os.path.join(tgt_dir, f"{i}-{j}.wav"), segment, sample_rate)
            final_segments.append([seg_start / sample_rate, seg_end / sample_rate])
            
    return final_segments


def main(rank, args, task_lines):
    num_gpus = torch.cuda.device_count()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank%num_gpus)
    world_size = args.world_size
    from funasr import AutoModel

    #device = torch.device(f"cuda:{rank%num_gpus}")
    device = torch.device("cpu")
    model = AutoModel(
        model="fsmn-vad",
        model_revision="v2.0.4",
        device="cpu",
        max_end_silence_time=500,
    )
    # model.to(device)

    done_tasks = []
    save_dir = args.save_dir
    
    # Check if we are in SCP mode or Task Dir mode
    is_scp_mode = args.in_scp is not None
    
    local_task_lines = task_lines[rank::world_size]
    
    output_segments = []

    for i, task in tqdm(enumerate(local_task_lines)):
        task_split = task.split()
        wav_file = task_split[0]
        
        if is_scp_mode:
            # In SCP mode, task is "rec_id wav_path"
            rec_id = task_split[0]
            wav_file = task_split[1]
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, rec_id)
        else:
            # In Task Dir mode, task is "wav_path [save_name]"
            if len(task_split) > 1:
                save_name = task_split[1]
            else:
                save_name = os.path.splitext(os.path.basename(wav_file))[0]
            save_path = os.path.join(save_dir, save_name)

        # Run VAD routine
        segments = routine(wav_file, save_path, model, device, args)
        
        if is_scp_mode and args.out_scp:
            for seg in segments:
                start_sec = seg[0]
                end_sec = seg[1]
                # Format: rec_id-start_end rec_id start end
                # But standard segments file is: unique_id rec_id start end
                # Let's use: rec_id-0000-0000 rec_id start end
                start_str = "{:06d}".format(int(start_sec * 100))
                end_str = "{:06d}".format(int(end_sec * 100))
                seg_id = f"{rec_id}-{start_str}-{end_str}"
                output_segments.append(f"{seg_id} {rec_id} {start_sec:.3f} {end_sec:.3f}")

        done_tasks.append(task)
        
        # Update done tasks for Task Dir mode
        if not is_scp_mode and args.task_dir:
            if i > 0 and i % args.done_update_interval == 0:
                with open(os.path.join(args.task_dir, f"done_{rank}"), "a") as f_done:
                    for line in done_tasks:
                        print(line, file=f_done)
                done_tasks = []

    # Final update
    if not is_scp_mode and args.task_dir:
        with open(os.path.join(args.task_dir, f"done_{rank}"), "a") as f_done:
            for line in done_tasks:
                print(line, file=f_done)
                
    # Write segments for SCP mode
    if is_scp_mode and args.out_scp:
        out_file = f"{args.out_scp}.{rank}"
        with open(out_file, "w") as f:
            for line in output_segments:
                f.write(line + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-duration", type=float, default=30)
    parser.add_argument("--merge-duration", type=float, default=10)
    parser.add_argument("--min-duration", type=float, default=2)
    parser.add_argument("--max-blank-duration", type=float, default=5)
    parser.add_argument("--task-dir", type=str)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--done-update-interval", type=int, default=100)
    
    # New arguments
    parser.add_argument("--in-scp", type=str, help="Input wav.scp file")
    parser.add_argument("--out-scp", type=str, help="Output segments file")

    args = parser.parse_args()
    
    task_lines = []
    
    if args.in_scp:
        # SCP Mode
        with open(args.in_scp, "r") as f:
            task_lines = f.read().splitlines()
    elif args.task_dir:
        # Task Dir Mode
        task_dir = args.task_dir
        if os.path.exists(os.path.join(task_dir, "running_task")):
             with open(os.path.join(task_dir, "running_task"), "r") as f_task:
                task_lines = f_task.read().splitlines()

        file_lis = os.listdir(task_dir)
        file_lis = [
            item
            for item in file_lis
            if os.path.isfile(os.path.join(task_dir, item)) and item.startswith("done")
        ]
        done_lines = []
        for item in file_lis:
            with open(os.path.join(task_dir, item), "r") as f_done:
                done_lines.extend(f_done.read().splitlines()) # Fixed expand -> extend

        task_lines = list(set(task_lines) - set(done_lines))
    else:
        print("Error: Either --in-scp or --task-dir must be provided.")
        exit(1)

    mp.set_start_method("spawn")

    processes = []
    for rank in range(args.world_size):
        p = mp.Process(target=main, args=(rank, args, task_lines))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    # Merge output segments if in SCP mode
    if args.in_scp and args.out_scp:
        with open(args.out_scp, "w") as f_out:
            for rank in range(args.world_size):
                part_file = f"{args.out_scp}.{rank}"
                if os.path.exists(part_file):
                    with open(part_file, "r") as f_part:
                        f_out.write(f_part.read())
                    os.remove(part_file)

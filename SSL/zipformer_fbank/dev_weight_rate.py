import os
import torch


checkpoint_home = "zipformer_fbank/exp-kmeans-ASR-500h-oracle-2000h-mask-065/exp-epoch-270-tri-stage-500h-lr-00005"

pre_idx = None
average_interval = 200
for i in range(159, 161):
    a = torch.load(os.path.join(checkpoint_home, f"epoch-{i}.pt"))
    batch_idx = a["batch_idx_train"]
    if pre_idx is not None:
        batch_n = batch_idx//average_interval
        pre_n = pre_idx//average_interval
        cur_avg_weight = batch_idx/(batch_idx-pre_idx)/batch_n
        pre_avg_weight = cur_avg_weight - pre_idx/(batch_idx-pre_idx)/pre_n
        print(f"{i}: {cur_avg_weight} {pre_avg_weight}")
        cur_avg_weight = batch_idx/(batch_idx-pre_idx)/batch_n
        print(f"{batch_idx/(batch_idx-pre_idx)*(batch_n-pre_n)/batch_n} {batch_idx/(batch_idx-pre_idx)*pre_n/batch_n-pre_idx/(batch_idx-pre_idx)}")
    pre_idx = batch_idx
    

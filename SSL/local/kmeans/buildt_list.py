import os

home_dir = "/workdir/work/icefall/egs/tencent/SSL/data"

pool_names = [str(i) for i in range(1, 5)] + [f"5_{i}" for i in range(9)]

result = []
def check(item, name):
    item = item.split(".")
    if len(item)>=3 and item[-1] == "gz" and item[-2] == "jsonl" and item[0] == name:
        return True
    return False

for pool_name in pool_names:
    pool_path = os.path.join(home_dir, f"ssl_pool{pool_name}", f"pool{pool_name}_split")
    lis = os.listdir(pool_path)

    lis = [item for item in lis if check(item, f"gigaspeech2-ssl_cuts_pool{pool_name}")]
    for item in lis:
        item_split = item.split(".")
        item_split[0] = f"gigaspeech2-ssl_cuts_pool{pool_name}_kmeans_iter1_epoch3"
        src = os.path.join(home_dir, pool_path, item)
        tgt = os.path.join(home_dir, pool_path, ".".join(item_split))
        result.append(f"{src} {tgt}")

for i in range(4):
    with open(f"/workdir/work/icefall/egs/tencent/SSL/tem_data/kmeans_iter1_epoch3_list_{i}", 'w') as f:
        f.write("\n".join(result[i::4]))


# Another version
# task_lis = []
# for item in args.src_dir:
#     pool_name = os.path.basename(item)
#     pool_name = pool_name[4:]
#     pool_path = os.path.join(item, f"{pool_name}_split")
#     sub_cut_lis = os.listdir(pool_path)
#     sub_task_lis = [(item.replace("_raw", ""), item.replace("_raw", "_kmeans")) for item in sub_cut_lis if item.endswith("jsonl.gz") and item.find("_raw")>=0]
#     sub_task_lis = [(os.path.join(pool_path, src), os.path.join(pool_path, tgt)) for src, tgt in sub_task_lis]
#     task_lis.extend(sub_task_lis)
# with open(args.task_list, 'w') as f:
#     for src, tgt in task_lis:
#         print(f"{src} {tgt}", file=f)
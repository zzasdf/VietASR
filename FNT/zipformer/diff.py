import torch
import os, sys
from pathlib import Path
# import matplotlib.pyplot as plt
from collections import OrderedDict

def compare_encoder_params(checkpoint_a, checkpoint_b, model_key='encoder'):
    ckpt_a = torch.load(checkpoint_a, map_location='cuda')
    ckpt_b = torch.load(checkpoint_b, map_location='cuda')

    model_a = OrderedDict((k, v) for k, v in ckpt_a['model'].items() 
                          if k.startswith(model_key))
    model_b = OrderedDict((k, v) for k, v in ckpt_b['model'].items()
                          if k.startswith(model_key))
    # print('model load done')

    if set(model_a.keys()) != set(model_b.keys()):
        missing_a = set(model_b.keys()) - set(model_a.keys())
        missing_b = set(model_a.keys()) - set(model_b.keys())
        print(f"A missing: {missing_a}")
        print(f"B missing: {missing_b}")

    diff_report = []
    cosine_sims = []
    for name, param_a in model_a.items():
        # print(name)
        param_b = model_b[name]
        # abs_diff = torch.abs(param_a - param_b)
        # if abs_diff != 0:
        if not torch.all(param_a == param_b):
            print(f'{name} not equal!')

    return diff_report


if __name__ == "__main__":
    model_a = sys.argv[1]
    model_b = sys.argv[2]
    diff_results = compare_encoder_params(
        Path(model_a), 
        Path(model_b),
        model_key='encoder'
    )
    diff_results = compare_encoder_params(
        Path(model_a), 
        Path(model_b),
        model_key='blank'
    )
    diff_results = compare_encoder_params(
        Path(model_a), 
        Path(model_b),
        model_key='joiner'
    )
    diff_results = compare_encoder_params(
        Path(model_a), 
        Path(model_b),
        model_key='vocab'
    )
    
import logging

import torch
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)


def get_avg_checkpoint(
    exp_dir=None,
    epoch=1,
    avg=1,
    use_averaged_model=True,
    run_iter=0,
    device=torch.device("cpu"),
):
    if not use_averaged_model:
        if run_iter > 0:
            filenames = find_checkpoints(exp_dir, iteration=-run_iter)[:avg]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for" f" --iter {run_iter}, --avg {avg}"
                )
            elif len(filenames) < avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {run_iter}, --avg {avg}"
                )
            logging.info(f"averaging {filenames}")
            return average_checkpoints(filenames, device=device)
        elif avg == 1:
            return torch.load(f"{exp_dir}/epoch-{epoch}.pt", map_location=device)[
                "model"
            ]
        else:
            start = epoch - avg + 1
            filenames = []
            for i in range(start, epoch + 1):
                if i >= 1:
                    filenames.append(f"{exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            return average_checkpoints(filenames, device=device)
    else:
        if run_iter > 0:
            filenames = find_checkpoints(exp_dir, iteration=-run_iter)[: avg + 1]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for" f" --iter {run_iter}, --avg {avg}"
                )
            elif len(filenames) < avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {run_iter}, --avg {avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            return average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        else:
            assert avg > 0, avg
            start = epoch - avg
            assert start >= 1, start
            filename_start = f"{exp_dir}/epoch-{start}.pt"
            filename_end = f"{exp_dir}/epoch-{epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {epoch}"
            )
            return average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )

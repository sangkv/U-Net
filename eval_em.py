import json
import numpy as np
import tifffile as tiff

from em_metrics import (
    membrane_to_neuron_instances,
    rand_error,
    variation_of_information
)


def main():
    # ----- load data -----
    pred_mem = tiff.imread("test-binary.tif")   # {0,1}
    gt_mem = tiff.imread("data/train-labels.tif")  # {0,1}

    assert pred_mem.shape == gt_mem.shape

    # ----- convert to neuron instances -----
    pred_inst = membrane_to_neuron_instances(pred_mem)
    gt_inst = membrane_to_neuron_instances(gt_mem)

    # ----- metrics -----
    rand_err = rand_error(pred_inst, gt_inst)
    vi, vi_split, vi_merge = variation_of_information(pred_inst, gt_inst)

    result = {
        "rand_error": float(rand_err),
        "vi": float(vi),
        "vi_split": float(vi_split),
        "vi_merge": float(vi_merge)
    }

    with open("eval/em_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("EM Evaluation:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

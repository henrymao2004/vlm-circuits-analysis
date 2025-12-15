# Hack to avoid some import problem due to the library being a subfolder
import sys
import torch
import argparse
import logging
import os
from typing import List

try:
    sys.path.append("third_party/TransformerLens")
    import transformer_lens as lens  # Some python problem causes this to throw on the first import
except:
    import transformer_lens as lens

from vision_language_prompts import VLPrompt
from attr_patching import node_attribution_patching_ig
from evaluation_utils import circuit_faithfulness
from general_utils import (
    get_top_scoring_components,
    set_deterministic,
)
from analysis_utils import SUPPORTED_TASKS, load_model, load_dataset


METRICS = ["LD"]
PERCENTAGES = sorted(
    torch.arange(0.0, 0.21, 0.01).tolist()  # High res in low amounts of nodes
    + torch.arange(0.3, 1.01, 0.2).tolist()  # Low res in high amounts of nodes
    + [0.001, 0.005]  # Some extra points
)

torch.set_grad_enabled(False)
device = "cuda"


def analyze_faithfulness(
    model: lens.HookedVLTransformer,
    scores_path_template: str,
    eval_prompts: List[VLPrompt],
    args: dict,
):
    """
    Calculate the faithfulnes of each top-p% node-based circuit.
    """
    model.cfg.ungroup_grouped_query_attention = True

    for metric in METRICS:
        scores = torch.load(
            scores_path_template.format(metric=metric), weights_only=True
        )
        scores = {
            k: v.abs() for (k, v) in scores.items()
        }  # Attribution scores are abs-ed
        seq_len = scores[f"blocks.0.attn.hook_z"].shape[0]

        logging.info(f"Analysing faithfulness for metric {metric}")
        faithfulness_results_path = f"./data/{args.task_name}/results/{args.model_name}/faithfulness_{metric}_{'l' if args.language_only else 'vl'}_node_circuit.pt"
        if os.path.exists(faithfulness_results_path):
            percentages, faiths_cf_ablations, faiths_completed_mask = torch.load(
                faithfulness_results_path, weights_only=True
            )
        else:
            percentages = PERCENTAGES
            faiths_cf_ablations = torch.zeros(len(percentages), len(percentages))
            faiths_completed_mask = torch.zeros(
                len(percentages), len(percentages), dtype=torch.bool
            )

        sorted_heads, sorted_mlp_neurons = None, None
        # Verify we have values for the diagonal of the faithfulness matrix
        logging.info("Verifying calculation of diagonal values")
        for i, percent in enumerate(percentages):
            if faiths_completed_mask[i, i]:
                continue
            logging.info(f"{percent=}")
            n_mlp_neurons = int(
                percent * model.cfg.d_mlp * model.cfg.n_layers * seq_len
            )
            n_heads = int(percent * model.cfg.n_heads * model.cfg.n_layers * seq_len)

            top_heads, top_mlps_with_neurons, sorted_heads, sorted_mlp_neurons = (
                get_top_scoring_components(
                    model,
                    scores,
                    n_heads,
                    n_mlp_neurons,
                    sorted_heads,
                    sorted_mlp_neurons,
                )
            )
            circuit_comps = top_heads + top_mlps_with_neurons

            faiths_cf_ablations[i, i] = circuit_faithfulness(
                model, circuit_comps, eval_prompts, metric=metric
            )
            faiths_completed_mask[i, i] = True
            logging.info(f"Faithfulness: {faiths_cf_ablations[i, i] :.5f}")
            torch.save(
                (percentages, faiths_cf_ablations, faiths_completed_mask),
                faithfulness_results_path,
            )

        # logging.info("Starting off-diagonal calculations")
        # for i, mlp_percent in enumerate(percentages):
        #     for j, attn_percent in enumerate(percentages):
        #         if faiths_completed_mask[i, j]:
        #             continue
        #         logging.info(f"MLP: {mlp_percent}, Heads: {attn_percent}")
        #         n_mlp_neurons = int(
        #             mlp_percent * model.cfg.d_mlp * model.cfg.n_layers * seq_len
        #         )
        #         n_heads = int(
        #             attn_percent * model.cfg.n_heads * model.cfg.n_layers * seq_len
        #         )

        #         top_heads, top_mlps_with_neurons, sorted_heads, sorted_mlp_neurons = (
        #             get_top_scoring_components(
        #                 model,
        #                 scores,
        #                 n_heads,
        #                 n_mlp_neurons,
        #                 sorted_heads,
        #                 sorted_mlp_neurons,
        #             )
        #         )
        #         circuit_comps = top_heads + top_mlps_with_neurons

        #         faiths_cf_ablations[i, j] = circuit_faithfulness_nodes_per_pos(
        #             model, circuit_comps, eval_prompts, metric=metric
        #         )
        #         faiths_completed_mask[i, j] = True
        #         logging.info(f"Faithfulness: {faiths_cf_ablations[i, j] :.5f}")
        #         torch.save(
        #             (percentages, faiths_cf_ablations, faiths_completed_mask),
        #             faithfulness_results_path,
        #         )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to be loaded")
    parser.add_argument("--model_path", type=str, help="Path to the model to be loaded")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument(
        "--task_name",
        type=str,
        choices=SUPPORTED_TASKS,
        help="Name of the task to be localized",
    )
    parser.add_argument(
        "--language_only",
        action="store_true",
        help="Whether to use only textual prompts for the chosen task",
    )
    parser.add_argument(
        "--ap_ig_steps", type=int, help="Number of steps for EAP-IG", default=5
    )
    args = parser.parse_args()
    return args


def main():
    logging.info("Running script_node_circuit_discovery_and_eval.py")
    args = parse_args()

    set_deterministic(args.seed)

    # Load model
    logging.info(f"Loading model {args.model_name} from {args.model_path}")
    model, processor = load_model(
        args.model_name,
        args.model_path,
        device,
        use_tlens_wrapper=True,
        extra_hooks=True,
        torch_dtype=torch.float32,
    )
    logging.info("Model loaded")

    # Load dataset
    logging.info(f"Loading dataset for task {args.task_name}")
    original_vl_prompts, vl_prompts, eval_vl_prompts = load_dataset(
        model,
        processor,
        args.task_name,
        args.model_name,
        args.language_only,
        args.seed,
        DISCOVERY_EVAL_SPLIT_PERCENT,
    )

    # Run node attribution patching on prompts or load pre-calculated results
    nap_ig_results_path = f"./data/{args.task_name}/results/{args.model_name}/node_scores/nap_ig_{'l' if args.language_only else 'vl'}_ig={args.ap_ig_steps}_metric={{metric}}.pt"
    model.cfg.ungroup_grouped_query_attention = True
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    os.makedirs(os.path.dirname(nap_ig_results_path), exist_ok=True)
    for metric in METRICS:
        if not os.path.exists(nap_ig_results_path.format(metric=metric)):
            scores = node_attribution_patching_ig(
                model, vl_prompts, metric=metric, ig_steps=args.ap_ig_steps
            )
            torch.save(scores, nap_ig_results_path.format(metric=metric))

    # Faithfulness analysis across graph sizes
    logging.info("Analysing faithfulness")
    analyze_faithfulness(model, nap_ig_results_path, eval_vl_prompts, args)

    logging.info("Analysis complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

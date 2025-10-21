import sys
import torch
import argparse
import logging
import random

try:
    sys.path.append("third_party/TransformerLens")
    import transformer_lens as lens  # Some python problem causes this to throw on the first import
except:
    import transformer_lens as lens
from component import Component
from general_utils import (
    set_deterministic,
    get_top_scoring_components,
)
from analysis_utils import SUPPORTED_TASKS, load_dataset, load_l_vl_scores, load_model
from modality_alignment_utils import (
    POS_MAPPINGS,
    PositionMapping,
    convert_components_modality,
    get_image_positions,
    get_text_sequence_positions,
)

METRIC = "LD"
DISCOVERY_EVAL_SPLIT_PERCENT = 0.75

torch.set_grad_enabled(False)
device = "cuda"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to be loaded")
    parser.add_argument("--model_path", type=str, help="Path to the model to be loaded")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument(
        "--task_name",
        type=str,
        choices=SUPPORTED_TASKS,
        help="Name of the task to be analyzed",
    )
    parser.add_argument(
        "--l_circuit_percentage",
        type=float,
        help="For which percentage of components in the L circuit to analyze cross-modality functionallity",
    )
    parser.add_argument(
        "--vl_circuit_percentage",
        type=float,
        help="For which percentage of components in the VL circuit to analyze cross-modality functionallity",
    )
    args = parser.parse_args()
    return args


def get_intersection(set_l, set_vl, pos_mapping=None):
    """
    Get the intersection of two sets of components. If pos_mapping is None,
    all positions are assumed to be None and ignored.

    Args:
        set_l (set): Set of components forming the L circuit.
        set_vl (set): Set of components forming the VL circuit.
        pos_mapping (PositionMapping, optional): Mapping of positions between L and VL circuits. Defaults to None.
    Returns:
        tuple: Intersection in VL and L modalities (might be different due to non-bijective mapping between the modalities).
    """
    if pos_mapping is None:
        # Assuming all positions are None
        iou = (
            0
            if len(set_vl) == 0
            else len(set(set_l).intersection(set(set_vl)))
            / len(set(set_l).union(set(set_vl)))
        )
        return iou, iou  # Returning two values for compatability with the other case
    else:
        l_mapped_to_vl = convert_components_modality(set_l, pos_mapping, l_to_vl=True)
        intersection_in_vl = (
            0
            if len(set_vl) == 0
            else len(l_mapped_to_vl.intersection(set(set_vl)))
            / len(l_mapped_to_vl.union(set(set_vl)))
        )

        vl_mapped_to_l = convert_components_modality(set_vl, pos_mapping, l_to_vl=False)
        intersection_in_l = (
            0
            if len(set_l) == 0
            else len(vl_mapped_to_l.intersection(set(set_l)))
            / len(vl_mapped_to_l.union(set(set_l)))
        )

        return intersection_in_vl, intersection_in_l


def split_to_d_q_g(components, seq_len, is_language, args):
    """
    Split the components into D (data), Q (query) and G (generation) components, by position.

    Args:
        components (list): List of components to be split.
        seq_len (int): Length of the sequence.
        is_language (bool): Whether the components are from a language model or not.
        args (argparse.Namespace): Parsed command line arguments. Used to extract model name and task name.

    Returns:
        tuple: Three lists of components: D, Q and G.
    """
    if is_language:
        D_limits = get_text_sequence_positions(args.model_name, args.task_name)
    else:
        D_limits = get_image_positions(args.model_name, args.task_name)

    Q_limits = [D_limits[-1], seq_len - 1]

    is_heads = isinstance(components[0], Component)
    if is_heads:
        D_comps = [c for c in components if D_limits[0] <= c.pos < D_limits[1]]
        Q_comps = [c for c in components if Q_limits[0] <= c.pos < Q_limits[1]]
        G_comps = [c for c in components if c.pos == seq_len - 1]
    else:
        # MLP Neurons, c[1] is the position
        D_comps = [c for c in components if D_limits[0] <= c[1] < D_limits[1]]
        Q_comps = [c for c in components if Q_limits[0] <= c[1] < Q_limits[1]]
        G_comps = [c for c in components if c[1] == seq_len - 1]

    return D_comps, Q_comps, G_comps


def remove_pos_info(comps):
    """
    Remove the position information from the components.

    Args:
        comps (list): List of components (or layer_position_neuron tuples) to be processed.

    Returns:
        list: List of components (or lpn tuples) with position information removed.
    """

    if isinstance(comps[0], Component):
        return [
            Component(c.hook_name, c.layer, c.head_idx, c.neuron_indices) for c in comps
        ]
    else:
        return [(lpn[0], 0, lpn[2]) for lpn in comps]


def get_full_intersection_dict(
    model,
    l_scores,
    vl_scores,
    pos_mapping,
    args,
):
    """
    Get the full intersection dictionary for L and VL circuits.
    The returned values are later summed and averaged to generate the Normalized IoU results.

    Args:
        model (lens.Lens): The model to be analyzed.
        l_scores (dict): Scores for the L circuit.
        vl_scores (dict): Scores for the VL circuit.
        pos_mapping (PositionMapping): Mapping of positions between L and VL circuits.
        args (argparse.Namespace): Parsed command line arguments.
    """
    l_seq_len = l_scores[f"blocks.0.attn.hook_z"].shape[0]
    vl_seq_len = vl_scores[f"blocks.0.attn.hook_z"].shape[0]

    both_vl = False
    if pos_mapping is None:
        # NOTE: THIS IS A HACK. THIS ASSUMED BOTH SCORE DICTS ARE VL SCORES, AND ARE ALIGNED POSITIONALLY.
        assert l_seq_len == vl_seq_len
        both_vl = True
        pos_mapping = PositionMapping([(i, i) for i in range(l_seq_len)])

    n_l_heads = int(
        args.l_circuit_percentage * model.cfg.n_heads * model.cfg.n_layers * l_seq_len
    )
    n_l_mlp_neurons = int(
        args.l_circuit_percentage * model.cfg.d_mlp * model.cfg.n_layers * l_seq_len
    )
    n_vl_heads = int(
        args.vl_circuit_percentage * model.cfg.n_heads * model.cfg.n_layers * vl_seq_len
    )
    n_vl_mlp_neurons = int(
        args.vl_circuit_percentage * model.cfg.d_mlp * model.cfg.n_layers * vl_seq_len
    )

    # Sort the found heads and neurons for both modalities
    _, __, l_sorted_heads, l_sorted_neurons = get_top_scoring_components(
        model, l_scores, n_l_heads, n_l_mlp_neurons
    )
    _, __, vl_sorted_heads, vl_sorted_neurons = get_top_scoring_components(
        model, vl_scores, n_vl_heads, n_vl_mlp_neurons
    )

    # Get randomized lists of components for a random baseline
    set_deterministic(args.seed)
    l_random_heads = random.sample(l_sorted_heads, len(l_sorted_heads))[:n_l_heads]
    l_random_neurons = random.sample(l_sorted_neurons, len(l_sorted_neurons))[
        :n_l_mlp_neurons
    ]
    vl_random_heads = random.sample(vl_sorted_heads, len(vl_sorted_heads))[:n_vl_heads]
    vl_random_neurons = random.sample(vl_sorted_neurons, len(vl_sorted_neurons))[
        :n_vl_mlp_neurons
    ]

    l_circuit_heads = l_sorted_heads[:n_l_heads]
    vl_circuit_heads = vl_sorted_heads[:n_vl_heads]
    l_circuit_neurons = l_sorted_neurons[:n_l_mlp_neurons]
    vl_circuit_neurons = vl_sorted_neurons[:n_vl_mlp_neurons]

    # Validate all components are unique
    assert (
        len(l_circuit_heads) == len(set(l_circuit_heads))
        and len(vl_circuit_heads) == len(set(vl_circuit_heads))
        and len(l_circuit_neurons) == len(set(l_circuit_neurons))
        and len(vl_circuit_neurons) == len(set(vl_circuit_neurons))
    )

    # Get intersections
    vl_head_iou, l_head_iou = get_intersection(
        l_circuit_heads, vl_circuit_heads, pos_mapping
    )
    vl_neurons_iou, l_neurons_iou = get_intersection(
        l_circuit_neurons, vl_circuit_neurons, pos_mapping
    )

    # Random baseline
    vl_head_baseline, l_head_baseline = get_intersection(
        l_random_heads, vl_random_heads, pos_mapping
    )
    vl_neurons_baseline, l_neurons_baseline = get_intersection(
        l_random_neurons, vl_random_neurons, pos_mapping
    )

    # Get intersections separately for D (data=image/text sequence), Q(query=question) and G(generation=last position) positions
    l_D_heads, l_Q_heads, l_G_heads = split_to_d_q_g(
        l_circuit_heads, l_seq_len, True and not both_vl, args
    )

    l_D_neurons, l_Q_neurons, l_G_neurons = split_to_d_q_g(
        l_circuit_neurons, l_seq_len, True and not both_vl, args
    )

    vl_D_heads, vl_Q_heads, vl_G_heads = split_to_d_q_g(
        vl_circuit_heads, vl_seq_len, False, args
    )

    vl_D_neurons, vl_Q_neurons, vl_G_neurons = split_to_d_q_g(
        vl_circuit_neurons, vl_seq_len, False, args
    )

    l_D_head_iou, vl_D_head_iou = get_intersection(l_D_heads, vl_D_heads, pos_mapping)
    l_Q_head_iou, vl_Q_head_iou = get_intersection(l_Q_heads, vl_Q_heads, pos_mapping)
    l_G_head_iou, vl_G_head_iou = get_intersection(l_G_heads, vl_G_heads, pos_mapping)
    l_D_neurons_iou, vl_D_neurons_iou = get_intersection(
        l_D_neurons, vl_D_neurons, pos_mapping
    )
    l_Q_neurons_iou, vl_Q_neurons_iou = get_intersection(
        l_Q_neurons, vl_Q_neurons, pos_mapping
    )
    l_G_neurons_iou, vl_G_neurons_iou = get_intersection(
        l_G_neurons, vl_G_neurons, pos_mapping
    )

    # Get baseline intersections separately for D, Q and G positions
    l_baseline_D_heads, l_baseline_Q_heads, l_baseline_G_heads = split_to_d_q_g(
        l_random_heads, l_seq_len, True and not both_vl, args
    )

    l_baseline_D_neurons, l_baseline_Q_neurons, l_baseline_G_neurons = split_to_d_q_g(
        l_random_neurons, l_seq_len, True and not both_vl, args
    )

    vl_baseline_D_heads, vl_baseline_Q_heads, vl_baseline_G_heads = split_to_d_q_g(
        vl_random_heads, vl_seq_len, False, args
    )

    vl_baseline_D_neurons, vl_baseline_Q_neurons, vl_baseline_G_neurons = (
        split_to_d_q_g(vl_random_neurons, vl_seq_len, False, args)
    )

    l_D_head_baseline, vl_D_head_baseline = get_intersection(
        l_baseline_D_heads, vl_baseline_D_heads, pos_mapping
    )
    l_Q_head_baseline, vl_Q_head_baseline = get_intersection(
        l_baseline_Q_heads, vl_baseline_Q_heads, pos_mapping
    )
    l_G_head_baseline, vl_G_head_baseline = get_intersection(
        l_baseline_G_heads, vl_baseline_G_heads, pos_mapping
    )
    l_D_neurons_baseline, vl_D_neurons_baseline = get_intersection(
        l_baseline_D_neurons, vl_baseline_D_neurons, pos_mapping
    )
    l_Q_neurons_baseline, vl_Q_neurons_baseline = get_intersection(
        l_baseline_Q_neurons, vl_baseline_Q_neurons, pos_mapping
    )
    l_G_neurons_baseline, vl_G_neurons_baseline = get_intersection(
        l_baseline_G_neurons, vl_baseline_G_neurons, pos_mapping
    )

    l_D_heads_no_pos = remove_pos_info(l_D_heads)
    vl_D_heads_no_pos = remove_pos_info(vl_D_heads)
    l_D_neurons_no_pos = remove_pos_info(l_D_neurons)
    vl_D_neurons_no_pos = remove_pos_info(vl_D_neurons)
    l_baseline_D_heads_no_pos = remove_pos_info(l_baseline_D_heads)
    vl_baseline_D_heads_no_pos = remove_pos_info(vl_baseline_D_heads)
    l_baseline_D_neurons_no_pos = remove_pos_info(l_baseline_D_neurons)
    vl_baseline_D_neurons_no_pos = remove_pos_info(vl_baseline_D_neurons)

    l_D_head_iou_no_pos, vl_D_head_iou_no_pos = get_intersection(
        l_D_heads_no_pos, vl_D_heads_no_pos
    )
    l_D_neurons_iou_no_pos, vl_D_neurons_iou_no_pos = get_intersection(
        l_D_neurons_no_pos, vl_D_neurons_no_pos
    )
    l_D_head_baseline_no_pos, vl_D_head_baseline_no_pos = get_intersection(
        l_baseline_D_heads_no_pos, vl_baseline_D_heads_no_pos
    )
    l_D_neurons_baseline_no_pos, vl_D_neurons_baseline_no_pos = get_intersection(
        l_baseline_D_neurons_no_pos, vl_baseline_D_neurons_no_pos
    )

    # Print and save results
    result_dict = {
        "l_percent": args.l_circuit_percentage,
        "vl_percent": args.vl_circuit_percentage,
        "vl_head_iou": vl_head_iou,
        "l_head_iou": l_head_iou,
        "vl_mlp_iou": vl_neurons_iou,
        "l_mlp_iou": l_neurons_iou,
        "vl_head_baseline": vl_head_baseline,
        "vl_mlp_baseline": vl_neurons_baseline,
        "l_head_baseline": l_head_baseline,
        "l_mlp_baseline": l_neurons_baseline,
        "vl_D_head_iou": vl_D_head_iou,
        "l_D_head_iou": l_D_head_iou,
        "vl_Q_head_iou": vl_Q_head_iou,
        "l_Q_head_iou": l_Q_head_iou,
        "vl_G_head_iou": vl_G_head_iou,
        "l_G_head_iou": l_G_head_iou,
        "vl_D_neurons_iou": vl_D_neurons_iou,
        "l_D_neurons_iou": l_D_neurons_iou,
        "vl_Q_neurons_iou": vl_Q_neurons_iou,
        "l_Q_neurons_iou": l_Q_neurons_iou,
        "vl_G_neurons_iou": vl_G_neurons_iou,
        "l_G_neurons_iou": l_G_neurons_iou,
        "vl_D_head_baseline": vl_D_head_baseline,
        "l_D_head_baseline": l_D_head_baseline,
        "vl_Q_head_baseline": vl_Q_head_baseline,
        "l_Q_head_baseline": l_Q_head_baseline,
        "vl_G_head_baseline": vl_G_head_baseline,
        "l_G_head_baseline": l_G_head_baseline,
        "vl_D_neurons_baseline": vl_D_neurons_baseline,
        "l_D_neurons_baseline": l_D_neurons_baseline,
        "vl_Q_neurons_baseline": vl_Q_neurons_baseline,
        "l_Q_neurons_baseline": l_Q_neurons_baseline,
        "vl_G_neurons_baseline": vl_G_neurons_baseline,
        "l_G_neurons_baseline": l_G_neurons_baseline,
        "vl_D_head_iou_no_pos": vl_D_head_iou_no_pos,
        "l_D_head_iou_no_pos": l_D_head_iou_no_pos,
        "vl_D_neurons_iou_no_pos": vl_D_neurons_iou_no_pos,
        "l_D_neurons_iou_no_pos": l_D_neurons_iou_no_pos,
        "vl_D_head_baseline_no_pos": vl_D_head_baseline_no_pos,
        "l_D_head_baseline_no_pos": l_D_head_baseline_no_pos,
        "vl_D_neurons_baseline_no_pos": vl_D_neurons_baseline_no_pos,
        "l_D_neurons_baseline_no_pos": l_D_neurons_baseline_no_pos,
    }
    return result_dict


def analyze_circuit_intersections(model, args, pos_mapping):
    intersection_results_path = (
        f"./data/{args.task_name}/results/{args.model_name}/intersection_results.pt"
    )

    # Load L and VL node scores
    logging.info(f"Loading L and VL scores")
    l_scores, vl_scores = load_l_vl_scores(args.task_name, args.model_name, METRIC)
    result_dict = get_full_intersection_dict(
        model,
        l_scores,
        vl_scores,
        pos_mapping,
        args,
    )
    logging.info(result_dict)
    torch.save(result_dict, intersection_results_path)


def main():
    logging.info("Running script_node_intersection.py")
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

    # Load L and VL datasets
    logging.info(f"Loading dataset for task {args.task_name}")
    vl_prompts = load_dataset(
        model=model,
        processor=processor,
        task_name=args.task_name,
        model_name=args.model_name,
        language_only=False,
        seed=args.seed,
        train_test_split_ratio=DISCOVERY_EVAL_SPLIT_PERCENT,
    )[0]
    l_prompts = load_dataset(
        model=model,
        processor=processor,
        task_name=args.task_name,
        model_name=args.model_name,
        language_only=True,
        seed=args.seed,
        train_test_split_ratio=DISCOVERY_EVAL_SPLIT_PERCENT,
    )[0]

    # Analyze the intersection of heads and neurons per percentage of components
    logging.info("Analyzing intersection of components")
    pos_mapping = POS_MAPPINGS[f"{args.model_name[:4]}_{args.task_name}"]
    pos_mapping.assert_full_mapping(l_prompts[0], vl_prompts[0], model)
    analyze_circuit_intersections(model, args, pos_mapping)

    logging.info("Analysis complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

"""SG23 explicit spike features and scalable exact/approximate solvers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_sg18_affordance_weighted_krr as sg18  # noqa: E402
from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments import e3_sg21_episodic_edge_graph as sg21  # noqa: E402
from experiments import e3_sg22_plan_path_constraints as sg22  # noqa: E402
from experiments import e3_sg22r_seventh_fresh_confirmation as sg22r  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg23_spike_feature_solvers.json")
DEFAULT_SG22R_REFERENCE = Path(
    "results/e3_scan/e3_sg22r_seventh_fresh_confirmation.json"
)
SG22R_REFERENCE_SHA256 = (
    "1A75839740A7913E555FBEBD5EB462AA4C50D5324709B11F507A9FB607B7DB92"
)
SG22R_EXPERIMENT = "E3-SG22R seventh-fresh constrained matched confirmation"
DEFAULT_CACHE = sg22r.DEFAULT_CACHE
CACHE_SHA256 = (
    "2016BF42DF694FBE6F4EDCD81E21C03E09F4A92348BDDB8909DD0118A2565A5E"
)
DEFAULT_CORPUS = sg22r.DEFAULT_CORPUS
DEFAULT_STRESS_SIZES = (1024, 2048, 4096)
DEFAULT_THREAD_SWEEP = (1, 2, 4, 8, 16)
NYSTROM_RANKS = (32, 64, 128, 256)
FEATURE_GRAM_TOLERANCE = 1e-10
EXACT_SCORE_TOLERANCE = 1e-6
PCG_RELATIVE_TOLERANCE = 1e-8
ANN_FASTEST_TRAIN_SECONDS = 0.59160
AFFORDANCE_MASK_WIDTH = 8
MASK_SCALE = 1.0 / math.sqrt(float(AFFORDANCE_MASK_WIDTH))


FeatureSignature = Tuple[Tuple[int, ...], ...]


@dataclass
class ExplicitFeatureMatrix:
    matrix: torch.Tensor
    vocabulary: Tuple[FeatureSignature, ...]
    rows: Tuple[Mapping[FeatureSignature, float], ...]
    build_seconds: float
    vocabulary_sha256: str

    @property
    def row_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.matrix.values().numel())

    @property
    def logical_csr_bytes(self) -> int:
        return int(
            self.matrix.crow_indices().numel() * 8
            + self.matrix.col_indices().numel() * 8
            + self.matrix.values().numel() * 8
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _load_frozen_json(
    path: Path, expected_sha256: str, expected_experiment: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != expected_sha256:
        raise ValueError(
            f"reference SHA mismatch for {path}: {digest} != {expected_sha256}"
        )
    value = json.loads(payload)
    if expected_experiment is not None and value.get("experiment") != expected_experiment:
        raise ValueError(f"unexpected experiment in {path}")
    return value, digest


def _feature_row(
    keys: Sequence[int],
    phase: int,
    mask: Sequence[float],
    plan_current: int,
    plan_next: int,
    return_edge: int,
) -> Dict[FeatureSignature, float]:
    if len(keys) != 4:
        raise ValueError("SG23 feature keys must contain four categorical slots")
    base_terms = (
        (0, int(phase), int(keys[3])),
        (1, int(phase), int(keys[2]), int(keys[3])),
        (2, int(phase), int(keys[1]), int(keys[2]), int(keys[3])),
        (
            3,
            int(phase),
            int(keys[0]),
            int(keys[1]),
            int(keys[2]),
            int(keys[3]),
        ),
    )
    mask_terms = [((0,), 1.0)]
    mask_terms.extend(
        ((1, index), MASK_SCALE)
        for index, value in enumerate(mask)
        if float(value) > 0.5
    )
    return_terms = ((0,), (1, int(return_edge)))
    plan_terms = (
        (0,),
        (1, int(plan_current)),
        (2, int(plan_next)),
    )
    row: Dict[FeatureSignature, float] = {}
    for base in base_terms:
        for mask_key, mask_value in mask_terms:
            for return_key in return_terms:
                for plan_key in plan_terms:
                    signature = (base, mask_key, return_key, plan_key)
                    if signature in row:
                        raise AssertionError("explicit spike feature collision")
                    row[signature] = mask_value
    return row


def _feature_rows(states: Mapping[str, Any]) -> Tuple[Dict[FeatureSignature, float], ...]:
    keys = states["keys"].detach().cpu().tolist()
    phases = states["phases"].detach().cpu().tolist()
    masks = states["masks"].detach().cpu().tolist()
    plan_current = states["plan_current"].detach().cpu().tolist()
    plan_next = states["plan_next"].detach().cpu().tolist()
    return_edges = states["return_edges"].detach().cpu().tolist()
    return tuple(
        _feature_row(key, phase, mask, current, following, returned)
        for key, phase, mask, current, following, returned in zip(
            keys,
            phases,
            masks,
            plan_current,
            plan_next,
            return_edges,
        )
    )


def build_explicit_features(
    states: Mapping[str, Any],
    *,
    vocabulary: Optional[Sequence[FeatureSignature]] = None,
) -> ExplicitFeatureMatrix:
    started = time.perf_counter_ns()
    rows = _feature_rows(states)
    if vocabulary is None:
        resolved_vocabulary = tuple(
            sorted({signature for row in rows for signature in row})
        )
    else:
        resolved_vocabulary = tuple(vocabulary)
    vocabulary_index = {
        signature: index
        for index, signature in enumerate(resolved_vocabulary)
    }
    crow = [0]
    columns = []
    values = []
    for row in rows:
        entries = sorted(
            (
                (vocabulary_index[signature], value)
                for signature, value in row.items()
                if signature in vocabulary_index
            ),
            key=lambda item: item[0],
        )
        columns.extend(column for column, _ in entries)
        values.extend(value for _, value in entries)
        crow.append(len(columns))
    matrix = torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=torch.int64),
        torch.tensor(columns, dtype=torch.int64),
        torch.tensor(values, dtype=torch.float64),
        size=(len(rows), len(resolved_vocabulary)),
        dtype=torch.float64,
    )
    vocabulary_payload = json.dumps(
        [[list(component) for component in signature] for signature in resolved_vocabulary],
        separators=(",", ":"),
    ).encode("utf-8")
    return ExplicitFeatureMatrix(
        matrix=matrix,
        vocabulary=resolved_vocabulary,
        rows=rows,
        build_seconds=(time.perf_counter_ns() - started) / 1e9,
        vocabulary_sha256=_sha256_bytes(vocabulary_payload),
    )


def explicit_cross_kernel(
    query_rows: Sequence[Mapping[FeatureSignature, float]],
    prototype_rows: Sequence[Mapping[FeatureSignature, float]],
) -> torch.Tensor:
    result = torch.empty(
        len(query_rows), len(prototype_rows), dtype=torch.float64
    )
    for query_index, query in enumerate(query_rows):
        for prototype_index, prototype in enumerate(prototype_rows):
            if len(query) <= len(prototype):
                value = sum(
                    weight * prototype.get(signature, 0.0)
                    for signature, weight in query.items()
                )
            else:
                value = sum(
                    weight * query.get(signature, 0.0)
                    for signature, weight in prototype.items()
                )
            result[query_index, prototype_index] = value
    return result


def explicit_dense_gram(
    features: ExplicitFeatureMatrix, *, block_size: int = 64
) -> Tuple[torch.Tensor, float]:
    started = time.perf_counter_ns()
    matrix = features.matrix
    crow = matrix.crow_indices()
    columns = matrix.col_indices()
    values = matrix.values()
    row_count, feature_count = matrix.shape
    gram = torch.empty(row_count, row_count, dtype=torch.float64)
    for start in range(0, row_count, block_size):
        stop = min(start + block_size, row_count)
        dense_block = torch.zeros(
            stop - start, feature_count, dtype=torch.float64
        )
        for local, row_index in enumerate(range(start, stop)):
            left = int(crow[row_index].item())
            right = int(crow[row_index + 1].item())
            dense_block[local, columns[left:right]] = values[left:right]
        gram[:, start:stop] = torch.sparse.mm(matrix, dense_block.T)
    return gram, (time.perf_counter_ns() - started) / 1e9


def _subset_states(
    states: Mapping[str, Any], indices: Sequence[int]
) -> Dict[str, Any]:
    index_tensor = torch.tensor(tuple(indices), dtype=torch.long)
    subset: Dict[str, Any] = {}
    row_count = int(states["keys"].shape[0])
    for name, value in states.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == row_count:
            subset[name] = value.index_select(0, index_tensor)
        else:
            subset[name] = value
    return subset


def generate_stress_states(
    base: Mapping[str, Any], row_count: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if row_count <= 0:
        raise ValueError("stress row count must be positive")
    key_supports = [
        tuple(int(value) for value in torch.unique(base["keys"][:, slot]).tolist())
        for slot in range(4)
    ]
    phase_support = tuple(int(value) for value in torch.unique(base["phases"]).tolist())
    current_support = tuple(
        int(value) for value in torch.unique(base["plan_current"]).tolist()
    )
    next_support = tuple(
        int(value) for value in torch.unique(base["plan_next"]).tolist()
    )
    return_support = tuple(
        int(value) for value in torch.unique(base["return_edges"]).tolist()
    )
    mask_support_tensor = torch.unique(base["masks"], dim=0, sorted=True)
    mask_support = tuple(
        tuple(float(value) for value in row)
        for row in mask_support_tensor.tolist()
    )
    supports: Tuple[Sequence[Any], ...] = (
        return_support,
        next_support,
        current_support,
        mask_support,
        phase_support,
        key_supports[3],
        key_supports[2],
        key_supports[1],
        key_supports[0],
    )
    capacity = math.prod(len(support) for support in supports)
    if capacity < row_count:
        raise ValueError(f"stress support capacity {capacity} < {row_count}")
    keys = []
    phases = []
    masks = []
    plan_current = []
    plan_next = []
    return_edges = []
    for row_index in range(row_count):
        quotient = row_index
        chosen = []
        for support in supports:
            chosen.append(support[quotient % len(support)])
            quotient //= len(support)
        returned, following, current, mask, phase, candidate, last1, last2, last3 = chosen
        return_edges.append(returned)
        plan_next.append(following)
        plan_current.append(current)
        masks.append(mask)
        phases.append(phase)
        keys.append((last3, last2, last1, candidate))
    stress = {
        "keys": torch.tensor(keys, dtype=torch.long),
        "phases": torch.tensor(phases, dtype=torch.long),
        "masks": torch.tensor(masks, dtype=torch.float64),
        "plan_current": torch.tensor(plan_current, dtype=torch.long),
        "plan_next": torch.tensor(plan_next, dtype=torch.long),
        "return_edges": torch.tensor(return_edges, dtype=torch.long),
        "counts": base["counts"][
            torch.arange(row_count, dtype=torch.long) % base["counts"].shape[0]
        ].clone(),
        "target_means": base["target_means"][
            torch.arange(row_count, dtype=torch.long)
            % base["target_means"].shape[0]
        ].clone(),
        "ambiguous_unique_key_count": 0,
    }
    combined = torch.cat(
        (
            stress["keys"],
            stress["phases"][:, None],
            stress["masks"].to(torch.long),
            stress["plan_current"][:, None],
            stress["plan_next"][:, None],
            stress["return_edges"][:, None],
        ),
        dim=1,
    )
    unique_count = int(torch.unique(combined, dim=0).shape[0])
    if unique_count != row_count:
        raise AssertionError("stress generator did not create unique states")
    fingerprint = hashlib.sha256()
    for name in (
        "keys",
        "phases",
        "masks",
        "plan_current",
        "plan_next",
        "return_edges",
        "counts",
        "target_means",
    ):
        fingerprint.update(name.encode("ascii"))
        fingerprint.update(stress[name].contiguous().numpy().tobytes())
    audit = {
        "row_count": row_count,
        "unique_state_count": unique_count,
        "support_capacity": capacity,
        "mask_support_count": len(mask_support),
        "canonical_sha256": fingerprint.hexdigest().upper(),
    }
    return stress, audit


def dense_weighted_cholesky(
    states: Mapping[str, Any]
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor]:
    kernel_started = time.perf_counter_ns()
    kernel = sg19.plan_edge_kernel(states, states)
    kernel_seconds = (time.perf_counter_ns() - kernel_started) / 1e9
    sqrt_counts = states["counts"].sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + sg19.FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], dtype=torch.float64)
    )
    rhs = sqrt_counts[:, None] * states["target_means"]
    solve_started = time.perf_counter_ns()
    factor = torch.linalg.cholesky(system)
    dual = torch.cholesky_solve(rhs, factor)
    coefficients = sqrt_counts[:, None] * dual
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    return coefficients, {
        "kernel_seconds": kernel_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": kernel_seconds + solve_seconds,
        "dense_kernel_logical_bytes": int(kernel.numel() * kernel.element_size()),
        "factor_logical_bytes": int(factor.numel() * factor.element_size()),
    }, kernel, system


def _feature_system_matvec(
    features: ExplicitFeatureMatrix,
    sqrt_counts: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    scaled = sqrt_counts[:, None] * vectors
    feature_projection = torch.sparse.mm(features.matrix.transpose(0, 1), scaled)
    reconstructed = torch.sparse.mm(features.matrix, feature_projection)
    return sqrt_counts[:, None] * reconstructed + sg19.FROZEN_LAMBDA * vectors


def _feature_system_diagonal(
    features: ExplicitFeatureMatrix, counts: torch.Tensor
) -> torch.Tensor:
    matrix = features.matrix
    lengths = matrix.crow_indices()[1:] - matrix.crow_indices()[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(features.row_count, dtype=torch.long), lengths
    )
    row_norms = torch.zeros(features.row_count, dtype=torch.float64)
    row_norms.index_add_(0, row_indices, matrix.values().square())
    return counts * row_norms + sg19.FROZEN_LAMBDA


def _block_preconditioner(
    system: Optional[torch.Tensor], states: Mapping[str, Any]
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Dict[str, Any]]:
    started = time.perf_counter_ns()
    groups: Dict[Tuple[int, int], list[int]] = defaultdict(list)
    for index, (phase, returned) in enumerate(
        zip(states["phases"].tolist(), states["return_edges"].tolist())
    ):
        groups[(int(phase), int(returned))].append(index)
    factors = []
    logical_bytes = 0
    kernel_seconds = 0.0
    for key in sorted(groups):
        indices = torch.tensor(groups[key], dtype=torch.long)
        if system is None:
            block_states = _subset_states(states, groups[key])
            kernel_started = time.perf_counter_ns()
            block_kernel = sg19.plan_edge_kernel(block_states, block_states)
            kernel_seconds += (time.perf_counter_ns() - kernel_started) / 1e9
            block_sqrt = block_states["counts"].sqrt()
            block = (
                block_sqrt[:, None]
                * block_kernel
                * block_sqrt[None, :]
                + sg19.FROZEN_LAMBDA
                * torch.eye(len(groups[key]), dtype=torch.float64)
            )
        else:
            block = system.index_select(0, indices).index_select(1, indices)
        factor = torch.linalg.cholesky(block)
        logical_bytes += int(factor.numel() * factor.element_size())
        factors.append((indices, factor))

    def apply(residual: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(residual)
        for indices, factor in factors:
            result.index_copy_(
                0,
                indices,
                torch.cholesky_solve(residual.index_select(0, indices), factor),
            )
        return result

    return apply, {
        "group_count": len(factors),
        "maximum_group_size": max(len(indices) for indices in groups.values()),
        "setup_seconds": (time.perf_counter_ns() - started) / 1e9,
        "kernel_seconds_within_setup": kernel_seconds,
        "constructed_without_full_system": system is None,
        "logical_bytes": logical_bytes,
    }


def _kernel_diagonal(states: Mapping[str, Any]) -> torch.Tensor:
    """Diagonal of the frozen strict-phase plan-edge kernel."""

    active_mask_bits = states["masks"].to(torch.float64).sum(dim=1)
    # Four strict phase×suffix atoms, return diagonal factor two, plan factor three.
    return 24.0 * (1.0 + active_mask_bits / AFFORDANCE_MASK_WIDTH)


def pivoted_cholesky_from_states(
    states: Mapping[str, Any], *, maximum_rank: int
) -> Tuple[torch.Tensor, Tuple[int, ...], Dict[str, Any]]:
    row_count = int(states["keys"].shape[0])
    maximum_rank = min(maximum_rank, row_count)
    factor = torch.zeros(row_count, maximum_rank, dtype=torch.float64)
    residual_diagonal = _kernel_diagonal(states)
    selected = torch.zeros(row_count, dtype=torch.bool)
    pivots = []
    kernel_seconds = 0.0
    started = time.perf_counter_ns()
    for column_index in range(maximum_rank):
        candidates = residual_diagonal.masked_fill(selected, -torch.inf)
        pivot = int(torch.argmax(candidates).item())
        pivot_residual = float(candidates[pivot].item())
        if pivot_residual <= 1e-14:
            break
        selected[pivot] = True
        pivots.append(pivot)
        kernel_started = time.perf_counter_ns()
        column_kernel = sg19.plan_edge_kernel(
            states, _subset_states(states, (pivot,))
        )[:, 0]
        kernel_seconds += (time.perf_counter_ns() - kernel_started) / 1e9
        if column_index:
            correction = factor[:, :column_index] @ factor[pivot, :column_index]
        else:
            correction = torch.zeros(row_count, dtype=torch.float64)
        column = (column_kernel - correction) / math.sqrt(pivot_residual)
        if column_index:
            column[
                torch.tensor(pivots[:-1], dtype=torch.long)
            ] = 0.0
        column[pivot] = math.sqrt(pivot_residual)
        factor[:, column_index] = column
        residual_diagonal = (residual_diagonal - column.square()).clamp_min(0.0)
        residual_diagonal[pivot] = 0.0
    effective_rank = len(pivots)
    factor = factor[:, :effective_rank]
    return factor, tuple(pivots), {
        "requested_rank": maximum_rank,
        "effective_rank": effective_rank,
        "kernel_column_seconds": kernel_seconds,
        "setup_seconds": (time.perf_counter_ns() - started) / 1e9,
        "maximum_residual_diagonal": float(residual_diagonal.max().item()),
    }


def _spectral_preconditioner(
    states: Mapping[str, Any], *, rank: int
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Dict[str, Any]]:
    full_setup_started = time.perf_counter_ns()
    factor, pivots, pivot_metrics = pivoted_cholesky_from_states(
        states, maximum_rank=rank
    )
    weighted_factor = states["counts"].sqrt()[:, None] * factor
    reflectors, tau = torch.geqrf(weighted_factor)
    effective_rank = int(weighted_factor.shape[1])
    triangular = torch.triu(reflectors[:effective_rank, :])
    subspace_system = (
        triangular @ triangular.T
        + sg19.FROZEN_LAMBDA
        * torch.eye(effective_rank, dtype=torch.float64)
    )
    subspace_factor = torch.linalg.cholesky(subspace_system)

    def apply(residual: torch.Tensor) -> torch.Tensor:
        coordinates = torch.ormqr(
            reflectors,
            tau,
            residual,
            left=True,
            transpose=True,
        )
        solved = coordinates.clone()
        solved[:effective_rank] = torch.cholesky_solve(
            coordinates[:effective_rank], subspace_factor
        )
        solved[effective_rank:] = (
            coordinates[effective_rank:] / sg19.FROZEN_LAMBDA
        )
        return torch.ormqr(
            reflectors,
            tau,
            solved,
            left=True,
            transpose=False,
        )

    logical_bytes = int(
        reflectors.numel() * reflectors.element_size()
        + tau.numel() * tau.element_size()
        + subspace_factor.numel() * subspace_factor.element_size()
    )
    return apply, {
        **pivot_metrics,
        "setup_seconds": (time.perf_counter_ns() - full_setup_started) / 1e9,
        "pivots_sha256": _sha256_bytes(json.dumps(pivots).encode("utf-8")),
        "inverse_implementation": "implicit_householder_qr_subspace_plus_lambda_complement",
        "fixed_iterative_refinement_steps": 0,
        "minimum_abs_r_diagonal": float(
            torch.diagonal(triangular).abs().min().item()
        ),
        "maximum_abs_r_diagonal": float(
            torch.diagonal(triangular).abs().max().item()
        ),
        "logical_bytes": logical_bytes,
    }


def spectral_iterative_refinement(
    features: ExplicitFeatureMatrix,
    states: Mapping[str, Any],
    *,
    rank: int,
    relative_tolerance: float = PCG_RELATIVE_TOLERANCE,
    maximum_refinements: int = 8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    apply_approximate_inverse, preconditioner_metrics = (
        _spectral_preconditioner(states, rank=rank)
    )
    sqrt_counts = states["counts"].sqrt()
    rhs = sqrt_counts[:, None] * states["target_means"]
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
    solve_started = time.perf_counter_ns()
    solution = apply_approximate_inverse(rhs)
    relative_history = []
    refinements = 0
    for refinement in range(maximum_refinements + 1):
        residual = rhs - _feature_system_matvec(
            features, sqrt_counts, solution
        )
        relative = torch.linalg.vector_norm(residual, dim=0) / rhs_norm
        maximum_relative = float(relative.max().item())
        relative_history.append(maximum_relative)
        if maximum_relative <= relative_tolerance:
            break
        if refinement == maximum_refinements:
            break
        solution = solution + apply_approximate_inverse(residual)
        refinements += 1
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    coefficients = sqrt_counts[:, None] * solution
    return coefficients, {
        "rank": rank,
        "effective_rank": preconditioner_metrics["effective_rank"],
        "relative_tolerance": relative_tolerance,
        "maximum_refinements": maximum_refinements,
        "refinements": refinements,
        "relative_residual_history": tuple(relative_history),
        "final_max_relative_residual": relative_history[-1],
        "converged": bool(relative_history[-1] <= relative_tolerance),
        "preconditioner": preconditioner_metrics,
        "solve_seconds": solve_seconds,
        "total_seconds": float(
            preconditioner_metrics["setup_seconds"] + solve_seconds
        ),
        "solver_state_logical_bytes": int(
            preconditioner_metrics["logical_bytes"]
            + 4 * rhs.numel() * rhs.element_size()
        ),
    }


def primal_pcg(
    features: ExplicitFeatureMatrix,
    states: Mapping[str, Any],
    *,
    relative_tolerance: float = PCG_RELATIVE_TOLERANCE,
    max_iterations: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    feature_count = features.feature_count
    if max_iterations is None:
        max_iterations = min(4 * features.row_count, 4096)
    counts = states["counts"]
    weighted_targets = counts[:, None] * states["target_means"]
    rhs = torch.sparse.mm(features.matrix.transpose(0, 1), weighted_targets)

    matrix = features.matrix
    lengths = matrix.crow_indices()[1:] - matrix.crow_indices()[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(features.row_count, dtype=torch.long), lengths
    )
    diagonal = torch.full(
        (feature_count,), sg19.FROZEN_LAMBDA, dtype=torch.float64
    )
    diagonal.index_add_(
        0,
        matrix.col_indices(),
        counts.index_select(0, row_indices) * matrix.values().square(),
    )

    def matvec(vectors: torch.Tensor) -> torch.Tensor:
        examples = torch.sparse.mm(matrix, vectors)
        return (
            torch.sparse.mm(
                matrix.transpose(0, 1), counts[:, None] * examples
            )
            + sg19.FROZEN_LAMBDA * vectors
        )

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    transformed = residual / diagonal[:, None]
    direction = transformed.clone()
    rz = (residual * transformed).sum(dim=0)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
    relative = torch.linalg.vector_norm(residual, dim=0) / rhs_norm
    solve_started = time.perf_counter_ns()
    iterations = 0
    residual_replacements = 0
    for iteration in range(1, max_iterations + 1):
        active = relative > relative_tolerance
        if not bool(active.any()):
            break
        system_direction = matvec(direction)
        denominator = (direction * system_direction).sum(dim=0)
        alpha = torch.where(
            active,
            rz / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(rz),
        )
        solution = solution + direction * alpha
        residual = residual - system_direction * alpha
        relative = torch.linalg.vector_norm(residual, dim=0) / rhs_norm
        iterations = iteration
        if float(relative.max().item()) <= relative_tolerance:
            true_residual = rhs - matvec(solution)
            true_relative = (
                torch.linalg.vector_norm(true_residual, dim=0) / rhs_norm
            )
            if float(true_relative.max().item()) <= relative_tolerance:
                residual = true_residual
                relative = true_relative
                break
            residual_replacements += 1
            residual = true_residual
            relative = true_relative
            transformed = residual / diagonal[:, None]
            direction = transformed.clone()
            rz = (residual * transformed).sum(dim=0)
            continue
        next_transformed = residual / diagonal[:, None]
        next_rz = (residual * next_transformed).sum(dim=0)
        next_active = relative > relative_tolerance
        beta = torch.where(
            next_active,
            next_rz / rz.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(rz),
        )
        direction = next_transformed + direction * beta
        transformed = next_transformed
        rz = next_rz
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    true_residual = rhs - matvec(solution)
    final_relative = torch.linalg.vector_norm(true_residual, dim=0) / rhs_norm
    return solution, {
        "iterations": iterations,
        "maximum_iterations": max_iterations,
        "relative_tolerance": relative_tolerance,
        "final_max_relative_residual": float(final_relative.max().item()),
        "converged": bool(float(final_relative.max().item()) <= relative_tolerance),
        "residual_replacement_count": residual_replacements,
        "solve_seconds": solve_seconds,
        "total_seconds": solve_seconds,
        "feature_weight_logical_bytes": int(
            solution.numel() * solution.element_size()
        ),
        "solver_state_logical_bytes": int(
            5 * solution.numel() * solution.element_size()
            + diagonal.numel() * diagonal.element_size()
        ),
        "minimum_diagonal": float(diagonal.min().item()),
        "maximum_diagonal": float(diagonal.max().item()),
    }


def block_pcg(
    features: ExplicitFeatureMatrix,
    states: Mapping[str, Any],
    *,
    preconditioner: str,
    dense_system: Optional[torch.Tensor] = None,
    relative_tolerance: float = PCG_RELATIVE_TOLERANCE,
    max_iterations: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    row_count = features.row_count
    if max_iterations is None:
        max_iterations = min(4 * row_count, 4096)
    sqrt_counts = states["counts"].sqrt()
    rhs = sqrt_counts[:, None] * states["target_means"]
    diagonal = _feature_system_diagonal(features, states["counts"])
    setup_started = time.perf_counter_ns()
    if preconditioner == "none":
        apply_preconditioner = lambda residual: residual
        preconditioner_metrics = {"logical_bytes": 0, "setup_seconds": 0.0}
    elif preconditioner == "jacobi":
        apply_preconditioner = lambda residual: residual / diagonal[:, None]
        preconditioner_metrics = {
            "logical_bytes": int(diagonal.numel() * diagonal.element_size()),
            "setup_seconds": (time.perf_counter_ns() - setup_started) / 1e9,
            "minimum_diagonal": float(diagonal.min().item()),
            "maximum_diagonal": float(diagonal.max().item()),
        }
    elif preconditioner == "return_phase_block":
        apply_preconditioner, preconditioner_metrics = _block_preconditioner(
            dense_system, states
        )
    elif preconditioner.startswith("spectral_"):
        rank = int(preconditioner.split("_", maxsplit=1)[1])
        apply_preconditioner, preconditioner_metrics = _spectral_preconditioner(
            states, rank=rank
        )
    else:
        raise ValueError(f"unknown PCG preconditioner {preconditioner}")
    setup_seconds = float(preconditioner_metrics["setup_seconds"])

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    transformed = apply_preconditioner(residual)
    direction = transformed.clone()
    rz = (residual * transformed).sum(dim=0)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
    relative = torch.linalg.vector_norm(residual, dim=0) / rhs_norm
    initial_max_relative = float(relative.max().item())
    iterations = 0
    solve_started = time.perf_counter_ns()
    for iteration in range(1, max_iterations + 1):
        active = relative > relative_tolerance
        if not bool(active.any()):
            break
        system_direction = _feature_system_matvec(
            features, sqrt_counts, direction
        )
        denominator = (direction * system_direction).sum(dim=0)
        alpha = torch.where(
            active,
            rz / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(rz),
        )
        solution = solution + direction * alpha
        residual = residual - system_direction * alpha
        relative = torch.linalg.vector_norm(residual, dim=0) / rhs_norm
        iterations = iteration
        if float(relative.max().item()) <= relative_tolerance:
            true_residual = rhs - _feature_system_matvec(
                features, sqrt_counts, solution
            )
            true_relative = (
                torch.linalg.vector_norm(true_residual, dim=0) / rhs_norm
            )
            if float(true_relative.max().item()) <= relative_tolerance:
                residual = true_residual
                relative = true_relative
                break
            residual = true_residual
            relative = true_relative
            transformed = apply_preconditioner(residual)
            direction = transformed.clone()
            rz = (residual * transformed).sum(dim=0)
            continue
        next_transformed = apply_preconditioner(residual)
        next_rz = (residual * next_transformed).sum(dim=0)
        next_active = relative > relative_tolerance
        beta = torch.where(
            next_active,
            next_rz / rz.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(rz),
        )
        direction = next_transformed + direction * beta
        transformed = next_transformed
        rz = next_rz
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    final_residual = rhs - _feature_system_matvec(
        features, sqrt_counts, solution
    )
    final_relative = torch.linalg.vector_norm(final_residual, dim=0) / rhs_norm
    coefficients = sqrt_counts[:, None] * solution
    return coefficients, {
        "preconditioner": preconditioner,
        "iterations": iterations,
        "maximum_iterations": max_iterations,
        "relative_tolerance": relative_tolerance,
        "initial_max_relative_residual": initial_max_relative,
        "final_max_relative_residual": float(final_relative.max().item()),
        "converged": bool(float(final_relative.max().item()) <= relative_tolerance),
        "preconditioner": {"name": preconditioner, **preconditioner_metrics},
        "solve_seconds": solve_seconds,
        "total_seconds": setup_seconds + solve_seconds,
        "solver_state_logical_bytes": int(
            5 * rhs.numel() * rhs.element_size()
            + int(preconditioner_metrics["logical_bytes"])
        ),
    }


def online_block_cholesky(
    states: Mapping[str, Any], *, block_size: int = 64
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    row_count = int(states["keys"].shape[0])
    sqrt_counts = states["counts"].sqrt()
    factor: Optional[torch.Tensor] = None
    kernel_seconds = 0.0
    factor_seconds = 0.0
    for start in range(0, row_count, block_size):
        stop = min(start + block_size, row_count)
        new_indices = tuple(range(start, stop))
        new_states = _subset_states(states, new_indices)
        kernel_started = time.perf_counter_ns()
        diagonal_kernel = sg19.plan_edge_kernel(new_states, new_states)
        if start:
            old_states = _subset_states(states, tuple(range(start)))
            cross_kernel = sg19.plan_edge_kernel(old_states, new_states)
        else:
            cross_kernel = None
        kernel_seconds += (time.perf_counter_ns() - kernel_started) / 1e9
        factor_started = time.perf_counter_ns()
        new_sqrt = sqrt_counts[start:stop]
        diagonal = (
            new_sqrt[:, None] * diagonal_kernel * new_sqrt[None, :]
            + sg19.FROZEN_LAMBDA
            * torch.eye(stop - start, dtype=torch.float64)
        )
        if factor is None:
            factor = torch.linalg.cholesky(diagonal)
        else:
            weighted_cross = (
                sqrt_counts[:start, None]
                * cross_kernel
                * new_sqrt[None, :]
            )
            projected = torch.linalg.solve_triangular(
                factor, weighted_cross, upper=False
            )
            schur = diagonal - projected.T @ projected
            schur = 0.5 * (schur + schur.T)
            schur_factor = torch.linalg.cholesky(schur)
            expanded = torch.zeros(stop, stop, dtype=torch.float64)
            expanded[:start, :start] = factor
            expanded[start:stop, :start] = projected.T
            expanded[start:stop, start:stop] = schur_factor
            factor = expanded
        factor_seconds += (time.perf_counter_ns() - factor_started) / 1e9
    if factor is None:
        raise AssertionError("online factor was not built")
    rhs = sqrt_counts[:, None] * states["target_means"]
    solve_started = time.perf_counter_ns()
    dual = torch.cholesky_solve(rhs, factor)
    coefficients = sqrt_counts[:, None] * dual
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    return coefficients, {
        "block_size": block_size,
        "block_count": math.ceil(row_count / block_size),
        "kernel_seconds": kernel_seconds,
        "factor_update_seconds": factor_seconds,
        "final_solve_seconds": solve_seconds,
        "total_seconds": kernel_seconds + factor_seconds + solve_seconds,
        "factor_logical_bytes": int(factor.numel() * factor.element_size()),
    }


def pivoted_cholesky(
    kernel: torch.Tensor, *, maximum_rank: int
) -> Tuple[torch.Tensor, Tuple[int, ...], Dict[int, float], Dict[str, Any]]:
    row_count = int(kernel.shape[0])
    maximum_rank = min(maximum_rank, row_count)
    factor = torch.zeros(row_count, maximum_rank, dtype=kernel.dtype)
    residual_diagonal = torch.diagonal(kernel).clone()
    selected = torch.zeros(row_count, dtype=torch.bool)
    pivots = []
    checkpoint_seconds: Dict[int, float] = {}
    started = time.perf_counter_ns()
    for column_index in range(maximum_rank):
        candidates = residual_diagonal.masked_fill(selected, -torch.inf)
        pivot = int(torch.argmax(candidates).item())
        pivot_residual = float(candidates[pivot].item())
        if pivot_residual <= 1e-14:
            break
        selected[pivot] = True
        pivots.append(pivot)
        denominator = math.sqrt(pivot_residual)
        if column_index:
            correction = factor[:, :column_index] @ factor[pivot, :column_index]
        else:
            correction = torch.zeros(row_count, dtype=kernel.dtype)
        column = (kernel[:, pivot] - correction) / denominator
        if column_index:
            column[
                torch.tensor(pivots[:-1], dtype=torch.long)
            ] = 0.0
        column[pivot] = denominator
        factor[:, column_index] = column
        residual_diagonal = (residual_diagonal - column.square()).clamp_min(0.0)
        residual_diagonal[pivot] = 0.0
        rank = column_index + 1
        if rank in NYSTROM_RANKS or rank == maximum_rank:
            checkpoint_seconds[rank] = (time.perf_counter_ns() - started) / 1e9
    effective_rank = len(pivots)
    factor = factor[:, :effective_rank]
    reconstruction_error = float(
        (kernel - factor @ factor.T).abs().max().item()
    )
    return factor, tuple(pivots), checkpoint_seconds, {
        "requested_maximum_rank": maximum_rank,
        "effective_rank": effective_rank,
        "reconstruction_max_abs_error_at_effective_rank": reconstruction_error,
        "maximum_residual_diagonal": float(residual_diagonal.max().item()),
    }


def low_rank_weighted_solution(
    factor: torch.Tensor,
    pivots: Sequence[int],
    states: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    rank = int(factor.shape[1])
    sqrt_counts = states["counts"].sqrt()
    rhs = sqrt_counts[:, None] * states["target_means"]
    weighted_factor = sqrt_counts[:, None] * factor
    solve_started = time.perf_counter_ns()
    small_system = (
        weighted_factor.T @ weighted_factor
        + sg19.FROZEN_LAMBDA * torch.eye(rank, dtype=torch.float64)
    )
    projected_rhs = weighted_factor.T @ rhs
    theta = torch.linalg.solve(small_system, projected_rhs)
    pivot_index = torch.tensor(tuple(pivots[:rank]), dtype=torch.long)
    pivot_factor = factor.index_select(0, pivot_index)
    transform = torch.linalg.solve_triangular(
        pivot_factor.T,
        torch.eye(rank, dtype=torch.float64),
        upper=True,
    )
    gamma = transform @ theta
    coefficients = torch.zeros(
        factor.shape[0], states["target_means"].shape[1], dtype=torch.float64
    )
    coefficients.index_copy_(0, pivot_index, gamma)
    relative_residual = torch.linalg.vector_norm(
        small_system @ theta - projected_rhs, dim=0
    ) / torch.linalg.vector_norm(projected_rhs, dim=0).clamp_min(1e-30)
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    return coefficients, gamma, {
        "rank": rank,
        "solve_seconds": solve_seconds,
        "small_system_logical_bytes": int(
            small_system.numel() * small_system.element_size()
        ),
        "maximum_small_system_relative_residual": float(
            relative_residual.max().item()
        ),
        "prediction_space_formula": "theta=(B.T@B+lambda*I)^-1@B.T@rhs",
        "coefficient_representation": "landmark_supported_prediction_equivalent",
        "avoids_dual_lambda_subtraction": True,
    }


class LowRankRuntimeScorer:
    def __init__(
        self,
        *,
        alphabet_index: Mapping[str, int],
        landmarks: Mapping[str, Any],
        gamma: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.alphabet_index = alphabet_index
        self.device = device
        self.landmarks = {
            name: value.to(
                device=device,
                dtype=(torch.float32 if name == "masks" else value.dtype),
            )
            for name, value in landmarks.items()
            if isinstance(value, torch.Tensor)
            and name
            in (
                "keys",
                "phases",
                "masks",
                "plan_current",
                "plan_next",
                "return_edges",
            )
        }
        self.gamma = gamma.to(device=device, dtype=torch.float32)

    def __call__(
        self,
        context_actions: Sequence[str],
        current_mask: Sequence[int],
        candidate_action: str,
        plan: Sequence[str],
        last_move: Optional[str],
    ) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
        phase = len(context_actions)
        plan_current, plan_next = sg19._plan_slots(plan, phase)
        pad_index = len(self.alphabet_index)
        query = {
            "keys": torch.tensor(
                [
                    sg18.sg13._padded_history_key(
                        context_actions,
                        candidate_action,
                        alphabet_index=self.alphabet_index,
                        pad_index=pad_index,
                    )
                ],
                dtype=torch.long,
                device=self.device,
            ),
            "phases": torch.tensor((phase,), dtype=torch.long, device=self.device),
            "masks": torch.tensor(
                [current_mask], dtype=torch.float32, device=self.device
            ),
            "plan_current": torch.tensor(
                (
                    pad_index
                    if plan_current == sg19.PLAN_PAD
                    else sg19.action_index(plan_current, self.alphabet_index),
                ),
                dtype=torch.long,
                device=self.device,
            ),
            "plan_next": torch.tensor(
                (
                    pad_index
                    if plan_next == sg19.PLAN_PAD
                    else sg19.action_index(plan_next, self.alphabet_index),
                ),
                dtype=torch.long,
                device=self.device,
            ),
            "return_edges": torch.tensor(
                (sg19.return_edge_spike(last_move, candidate_action),),
                dtype=torch.long,
                device=self.device,
            ),
        }
        started = time.perf_counter_ns()
        scores = (
            sg19.plan_edge_kernel(
                query, self.landmarks, dtype=torch.float32
            )
            @ self.gamma
        )
        scores.sum().item()
        predicted_mask = tuple(
            int(value)
            for value in (
                scores[0, sg18.NEXT_MASK_OFFSET :]
                > sg18.MASK_DECISION_THRESHOLD
            )
            .cpu()
            .tolist()
        )
        return scores, predicted_mask, (time.perf_counter_ns() - started) / 1e6


def _prediction_equivalence(
    first_scores: torch.Tensor, second_scores: torch.Tensor
) -> bool:
    return bool(
        torch.equal(
            sg10._prediction_matrix(first_scores[:, : sg10.TOTAL_LOGITS]),
            sg10._prediction_matrix(second_scores[:, : sg10.TOTAL_LOGITS]),
        )
        and torch.equal(
            first_scores[:, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD,
            second_scores[:, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD,
        )
    )


def _load_problem(args: argparse.Namespace) -> Dict[str, Any]:
    reference, reference_sha = _load_frozen_json(
        args.sg22r_reference.expanduser().resolve(),
        SG22R_REFERENCE_SHA256,
        SG22R_EXPERIMENT,
    )
    if reference["decision"]["overall"] != "PASS":
        raise ValueError("SG23 requires passing SG22R reference")
    cache, cache_sha = _load_frozen_json(
        args.cache.expanduser().resolve(), CACHE_SHA256
    )
    corpus_root = args.corpus_dir.expanduser().resolve()
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=sg22r.EXPECTED_SEEDS
    )
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    alphabet = build_action_alphabet(examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_order = sg18._action_order(corpus_root)
    plans, plan_audit = sg19.load_objective_plans(corpus_root)
    exhaustive = cache["exhaustive"]
    repaired_tree, repair_audit = sg17.repair_persistent_room_semantics(
        cache["branch_tree"]
    )
    if repair_audit["changed_pair_count"] != 0:
        raise AssertionError("SG23 frozen seventh tree unexpectedly changed")
    tensors = {
        split: sg19.tensorize_extended(
            exhaustive[split]["records"],
            plans[split],
            alphabet_index=alphabet_index,
            device=torch.device("cpu"),
        )
        for split in SPLITS
    }
    unique = sg19.compress_extended(tensors["train"])
    snapshots, graph_audit = sg21.build_graph_snapshots(
        corpus_root, corpus, exhaustive, action_order
    )
    constraint_audit = sg22.audit_plan_path_constraint(
        exhaustive, plans, action_order
    )
    return {
        "reference": reference,
        "reference_sha": reference_sha,
        "cache_sha": cache_sha,
        "manifest": manifest,
        "plan_audit": plan_audit,
        "corpus": corpus,
        "exhaustive": exhaustive,
        "tree": repaired_tree,
        "tensors": tensors,
        "unique": unique,
        "snapshots": snapshots,
        "graph_audit": graph_audit,
        "constraint_audit": constraint_audit,
        "alphabet_index": alphabet_index,
        "action_order": action_order,
        "plans": plans,
    }


def _evaluate_exact_route(
    problem: Mapping[str, Any], coefficients: torch.Tensor
) -> Dict[str, Any]:
    split_metrics = {
        split: sg21.evaluate_split_with_graph(
            split,
            problem["tensors"][split],
            problem["exhaustive"][split]["records"],
            problem["snapshots"][split],
            problem["unique"],
            coefficients,
            problem["action_order"],
            plans=problem["plans"][split],
            enforce_plan_path_constraint=True,
        )
        for split in SPLITS
    }
    rollout = sg21.evaluate_two_step_with_graph(
        problem["tree"],
        problem["exhaustive"]["test"]["records"],
        problem["snapshots"]["test"],
        problem["plans"]["test"],
        alphabet_index=problem["alphabet_index"],
        unique=problem["unique"],
        coefficients=coefficients,
        action_order=problem["action_order"],
        device=torch.device("cpu"),
        enforce_plan_path_constraint=True,
    )
    return {"split_metrics": split_metrics, "rollout": rollout}


def _quality_is_perfect(metrics: Mapping[str, Any]) -> bool:
    if "evaluation_error" in metrics.get("rollout", {}):
        return False
    return bool(
        all(
            metrics["split_metrics"][split]["delta"]["exact_vector_accuracy"]
            == 1.0
            and metrics["split_metrics"][split]["next_affordance"][
                "exact_mask_accuracy"
            ]
            == 1.0
            for split in SPLITS
        )
        and metrics["rollout"]["teacher_forced_second"][
            "exact_vector_accuracy"
        ]
        == 1.0
        and metrics["rollout"]["self_rollout_second"][
            "exact_vector_accuracy"
        ]
        == 1.0
    )


def _evaluate_low_rank(
    problem: Mapping[str, Any],
    factor: torch.Tensor,
    pivots: Sequence[int],
    rank: int,
    dense_coefficients: torch.Tensor,
) -> Dict[str, Any]:
    rank_factor = factor[:, :rank]
    rank_pivots = tuple(pivots[:rank])
    _coefficients, gamma, solve_metrics = low_rank_weighted_solution(
        rank_factor, rank_pivots, problem["unique"]
    )
    landmarks = _subset_states(problem["unique"], rank_pivots)
    split_metrics = {}
    score_differences = {}
    for split in SPLITS:
        exact_scores = (
            sg19.plan_edge_kernel(problem["tensors"][split], problem["unique"])
            @ dense_coefficients
        )
        approximate_scores = (
            sg19.plan_edge_kernel(problem["tensors"][split], landmarks) @ gamma
        )
        score_differences[split] = float(
            (approximate_scores - exact_scores).abs().max().item()
        )
        split_metrics[split] = sg21.evaluate_split_with_graph(
            split,
            problem["tensors"][split],
            problem["exhaustive"][split]["records"],
            problem["snapshots"][split],
            problem["unique"],
            dense_coefficients,
            problem["action_order"],
            plans=problem["plans"][split],
            enforce_plan_path_constraint=True,
            score_matrix=approximate_scores,
        )
    scorer = LowRankRuntimeScorer(
        alphabet_index=problem["alphabet_index"],
        landmarks=landmarks,
        gamma=gamma,
        device=torch.device("cpu"),
    )
    try:
        rollout = sg21.evaluate_two_step_with_graph(
            problem["tree"],
            problem["exhaustive"]["test"]["records"],
            problem["snapshots"]["test"],
            problem["plans"]["test"],
            alphabet_index=problem["alphabet_index"],
            unique=problem["unique"],
            coefficients=dense_coefficients,
            action_order=problem["action_order"],
            device=torch.device("cpu"),
            score_fn=scorer,
            enforce_plan_path_constraint=True,
        )
    except (AssertionError, ValueError) as error:
        rollout = {
            "evaluation_error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        }
    result = {
        "rank": rank,
        "solve": solve_metrics,
        "score_max_abs_difference_vs_exact": score_differences,
        "split_metrics": split_metrics,
        "rollout": rollout,
        "deployment_logical_bytes": int(
            rank
            * (
                problem["unique"]["keys"].shape[1]
                + 1
                + len(problem["action_order"])
                + 3
                + gamma.shape[1] * 4
            )
        ),
    }
    result["perfect_quality"] = _quality_is_perfect(result)
    return result


def _common_training_seconds(problem: Mapping[str, Any]) -> float:
    train = problem["tensors"]["train"]
    return float(
        train["elapsed_seconds"]
        + train["extended_state_encoding_seconds"]
        + problem["unique"]["elapsed_seconds"]
        + problem["graph_audit"]["splits"]["train"]["build_seconds"]
    )


def _run_stress_scale(
    base: Mapping[str, Any], row_count: int
) -> Dict[str, Any]:
    states, generation_audit = generate_stress_states(base, row_count)
    features = build_explicit_features(states)
    audit_count = min(row_count, 192)
    audit_states = _subset_states(states, tuple(range(audit_count)))
    audit_features = build_explicit_features(audit_states)
    explicit_audit = explicit_cross_kernel(
        audit_features.rows, audit_features.rows
    )
    analytic_audit = sg19.plan_edge_kernel(audit_states, audit_states)
    feature_error = float((explicit_audit - analytic_audit).abs().max().item())
    dense_coefficients, dense_metrics, kernel, system = dense_weighted_cholesky(
        states
    )
    dense_scores = kernel @ dense_coefficients
    pcg_variants = {}
    pcg_coefficients = {}
    for preconditioner in ("jacobi", "return_phase_block"):
        coefficients, metrics = block_pcg(
            features,
            states,
            preconditioner=preconditioner,
            dense_system=None,
            max_iterations=min(4 * row_count, 4096),
        )
        scores = kernel @ coefficients
        metrics["train_score_max_abs_difference"] = float(
            (dense_scores - scores).abs().max().item()
        )
        metrics["prediction_equivalent"] = _prediction_equivalence(
            dense_scores, scores
        )
        pcg_variants[preconditioner] = metrics
        pcg_coefficients[preconditioner] = coefficients
    refinement_coefficients, refinement_metrics = spectral_iterative_refinement(
        features,
        states,
        rank=max(NYSTROM_RANKS),
    )
    refinement_scores = kernel @ refinement_coefficients
    refinement_metrics["train_score_max_abs_difference"] = float(
        (dense_scores - refinement_scores).abs().max().item()
    )
    refinement_metrics["prediction_equivalent"] = _prediction_equivalence(
        dense_scores, refinement_scores
    )
    primal_weights, primal_metrics = primal_pcg(
        features,
        states,
        max_iterations=min(4 * row_count, 4096),
    )
    primal_scores = torch.sparse.mm(features.matrix, primal_weights)
    primal_metrics["train_score_max_abs_difference"] = float(
        (dense_scores - primal_scores).abs().max().item()
    )
    primal_metrics["prediction_equivalent"] = _prediction_equivalence(
        dense_scores, primal_scores
    )
    converged = [
        name
        for name, metrics in pcg_variants.items()
        if metrics["converged"]
        and metrics["train_score_max_abs_difference"] <= EXACT_SCORE_TOLERANCE
        and metrics["prediction_equivalent"]
    ]
    selected_pcg = (
        min(converged, key=lambda name: pcg_variants[name]["total_seconds"])
        if converged
        else None
    )
    refinement_exact = bool(
        refinement_metrics["converged"]
        and refinement_metrics["train_score_max_abs_difference"]
        <= EXACT_SCORE_TOLERANCE
        and refinement_metrics["prediction_equivalent"]
    )
    exact_candidates = []
    if selected_pcg is not None:
        exact_candidates.append(
            (f"pcg:{selected_pcg}", pcg_variants[selected_pcg])
        )
    if refinement_exact:
        exact_candidates.append(("spectral_refinement:256", refinement_metrics))
    primal_exact = bool(
        primal_metrics["converged"]
        and primal_metrics["train_score_max_abs_difference"]
        <= EXACT_SCORE_TOLERANCE
        and primal_metrics["prediction_equivalent"]
    )
    if primal_exact:
        exact_candidates.append(("primal_pcg:jacobi", primal_metrics))
    selected_exact_route, selected_metrics = (
        min(exact_candidates, key=lambda item: item[1]["total_seconds"])
        if exact_candidates
        else (None, pcg_variants["jacobi"])
    )
    score_difference = selected_metrics["train_score_max_abs_difference"]
    prediction_equivalent = selected_metrics["prediction_equivalent"]
    dense_bytes = int(
        kernel.numel() * kernel.element_size()
        + system.numel() * system.element_size()
    )
    matrix_free_bytes = int(
        features.logical_csr_bytes
        + selected_metrics["solver_state_logical_bytes"]
    )
    return {
        "generation": generation_audit,
        "feature_map": {
            "feature_count": features.feature_count,
            "nnz": features.nnz,
            "build_seconds": features.build_seconds,
            "vocabulary_sha256": features.vocabulary_sha256,
            "logical_csr_bytes": features.logical_csr_bytes,
            "audit_row_count": audit_count,
            "audit_gram_max_abs_error": feature_error,
        },
        "dense": dense_metrics,
        "pcg": pcg_variants,
        "selected_pcg": selected_pcg,
        "spectral_iterative_refinement": refinement_metrics,
        "primal_pcg": primal_metrics,
        "selected_exact_route": selected_exact_route,
        "exactness": {
            "train_score_max_abs_difference": score_difference,
            "prediction_equivalent": prediction_equivalent,
        },
        "memory": {
            "dense_kernel_plus_system_bytes": dense_bytes,
            "matrix_free_feature_plus_solver_bytes": matrix_free_bytes,
            "dense_over_matrix_free_ratio": dense_bytes / matrix_free_bytes,
        },
        "wall_speedup_dense_over_pcg": (
            dense_metrics["total_seconds"] / selected_metrics["total_seconds"]
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(args.threads)
    problem = _load_problem(args)
    unique = problem["unique"]
    explicit = build_explicit_features(unique)
    explicit_gram, explicit_gram_seconds = explicit_dense_gram(explicit)
    dense_coefficients, dense_metrics, analytic_kernel, dense_system = (
        dense_weighted_cholesky(unique)
    )
    feature_gram_error = float(
        (explicit_gram - analytic_kernel).abs().max().item()
    )

    pcg_results = {}
    pcg_coefficients = {}
    for preconditioner in ("none", "jacobi", "return_phase_block"):
        coefficients, metrics = block_pcg(
            explicit,
            unique,
            preconditioner=preconditioner,
            dense_system=(
                dense_system if preconditioner == "return_phase_block" else None
            ),
        )
        scores = analytic_kernel @ coefficients
        dense_scores = analytic_kernel @ dense_coefficients
        metrics["coefficient_max_abs_difference"] = float(
            (coefficients - dense_coefficients).abs().max().item()
        )
        metrics["train_score_max_abs_difference"] = float(
            (scores - dense_scores).abs().max().item()
        )
        metrics["prediction_equivalent"] = _prediction_equivalence(
            scores, dense_scores
        )
        pcg_results[preconditioner] = metrics
        pcg_coefficients[preconditioner] = coefficients

    converged_names = [
        name
        for name, metrics in pcg_results.items()
        if metrics["converged"]
        and metrics["train_score_max_abs_difference"] <= EXACT_SCORE_TOLERANCE
        and metrics["prediction_equivalent"]
    ]
    selected_pcg = (
        min(converged_names, key=lambda name: pcg_results[name]["total_seconds"])
        if converged_names
        else None
    )
    online_coefficients, online_metrics = online_block_cholesky(
        unique, block_size=args.online_block_size
    )
    dense_scores = analytic_kernel @ dense_coefficients
    online_scores = analytic_kernel @ online_coefficients
    online_metrics.update(
        {
            "coefficient_max_abs_difference": float(
                (online_coefficients - dense_coefficients).abs().max().item()
            ),
            "train_score_max_abs_difference": float(
                (online_scores - dense_scores).abs().max().item()
            ),
            "prediction_equivalent": _prediction_equivalence(
                online_scores, dense_scores
            ),
        }
    )

    if selected_pcg is None:
        selected_coefficients = dense_coefficients
    else:
        selected_coefficients = pcg_coefficients[selected_pcg]
    exact_quality = _evaluate_exact_route(problem, selected_coefficients)
    exact_quality["perfect_quality"] = _quality_is_perfect(exact_quality)

    maximum_rank = min(max(NYSTROM_RANKS), unique["keys"].shape[0])
    pivot_factor, pivots, rank_times, pivot_metrics = pivoted_cholesky(
        analytic_kernel, maximum_rank=maximum_rank
    )
    nystrom = {}
    for rank in NYSTROM_RANKS:
        if rank <= pivot_factor.shape[1]:
            result = _evaluate_low_rank(
                problem,
                pivot_factor,
                pivots,
                rank,
                dense_coefficients,
            )
            result["pivot_build_seconds"] = rank_times.get(rank)
            nystrom[str(rank)] = result

    stress = {
        str(size): _run_stress_scale(unique, size)
        for size in args.stress_sizes
    }
    common_seconds = _common_training_seconds(problem)
    selected_pcg_training = (
        None
        if selected_pcg is None
        else common_seconds
        + explicit.build_seconds
        + pcg_results[selected_pcg]["total_seconds"]
    )
    feature_math_gate = bool(
        feature_gram_error <= FEATURE_GRAM_TOLERANCE
        and all(
            result["feature_map"]["audit_gram_max_abs_error"]
            <= FEATURE_GRAM_TOLERANCE
            for result in stress.values()
        )
    )
    exact_solver_gate = bool(
        selected_pcg is not None
        and online_metrics["train_score_max_abs_difference"]
        <= EXACT_SCORE_TOLERANCE
        and online_metrics["prediction_equivalent"]
    )
    scale_gate = any(
        result["memory"]["dense_over_matrix_free_ratio"] >= 2.0
        and result["selected_exact_route"] is not None
        and result["exactness"]["train_score_max_abs_difference"]
        <= EXACT_SCORE_TOLERANCE
        for result in stress.values()
    )
    speed_gate = bool(
        selected_pcg_training is not None
        and selected_pcg_training < ANN_FASTEST_TRAIN_SECONDS
    )
    quality_gate = bool(exact_quality["perfect_quality"])
    overall = bool(
        feature_math_gate
        and exact_solver_gate
        and scale_gate
        and speed_gate
        and quality_gate
    )
    return {
        "experiment": "E3-SG23 explicit spike features and scalable solvers",
        "references": {
            "sg22r_sha256": problem["reference_sha"],
            "cache_sha256": problem["cache_sha"],
        },
        "protocol": {
            "threads": args.threads,
            "stress_sizes": tuple(args.stress_sizes),
            "thread_sweep": tuple(args.thread_sweep),
            "pcg_relative_tolerance": PCG_RELATIVE_TOLERANCE,
            "exact_score_tolerance": EXACT_SCORE_TOLERANCE,
            "feature_gram_tolerance": FEATURE_GRAM_TOLERANCE,
            "nystrom_ranks": NYSTROM_RANKS,
            "online_block_size": args.online_block_size,
        },
        "data": {
            "train_examples": int(problem["tensors"]["train"]["keys"].shape[0]),
            "unique_prototypes": int(unique["keys"].shape[0]),
            "ambiguous_unique_keys": unique["ambiguous_unique_key_count"],
            "graph_audit_pass": bool(
                problem["graph_audit"]["all_masks_match_exhaustive_cache"]
                and problem["graph_audit"][
                    "all_binding_steps_precede_snapshot_root"
                ]
                and problem["graph_audit"]["all_rooms_unique_and_present"]
                and problem["graph_audit"]["no_edge_conflicts"]
            ),
            "constraint_audit_pass": bool(
                problem["constraint_audit"]["all_targets_match"]
            ),
        },
        "real_443": {
            "feature_map": {
                "feature_count": explicit.feature_count,
                "nnz": explicit.nnz,
                "build_seconds": explicit.build_seconds,
                "explicit_gram_seconds": explicit_gram_seconds,
                "gram_max_abs_error": feature_gram_error,
                "vocabulary_sha256": explicit.vocabulary_sha256,
                "logical_csr_bytes": explicit.logical_csr_bytes,
            },
            "dense": dense_metrics,
            "pcg": pcg_results,
            "selected_pcg": selected_pcg,
            "selected_pcg_deployment_training_seconds": selected_pcg_training,
            "online_block_cholesky": online_metrics,
            "exact_quality": exact_quality,
            "pivoted_cholesky": {
                **pivot_metrics,
                "checkpoint_seconds": rank_times,
                "pivots_sha256": _sha256_bytes(
                    json.dumps(pivots).encode("utf-8")
                ),
            },
            "nystrom": nystrom,
            "common_encoding_compression_graph_seconds": common_seconds,
        },
        "stress": stress,
        "decision": {
            "feature_math_gate": feature_math_gate,
            "exact_solver_gate": exact_solver_gate,
            "real_quality_gate": quality_gate,
            "real_training_speed_gate": speed_gate,
            "scale_speed_or_memory_gate": scale_gate,
            "passing_nystrom_ranks": tuple(
                int(rank)
                for rank, result in nystrom.items()
                if result["perfect_quality"]
            ),
            "overall": "PASS" if overall else "FAIL",
            "next_route": (
                "sg24_raw_language_multimodal_closed_loop"
                if overall
                else "sg23_solver_failure_diagnostic"
            ),
        },
    }


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--sg22r-reference", type=Path, default=DEFAULT_SG22R_REFERENCE
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--stress-sizes",
        type=_parse_int_tuple,
        default=DEFAULT_STRESS_SIZES,
    )
    parser.add_argument(
        "--thread-sweep", type=_parse_int_tuple, default=DEFAULT_THREAD_SWEEP
    )
    parser.add_argument("--online-block-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.online_block_size <= 0:
        parser.error("threads and online block size must be positive")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Low-rank projection helpers for FE-2H.

REQ-005 requires a true low-rank replacement for dense projection layers.
This module implements a factorised linear layer that replaces the dense
matmul itself; additive LoRA-style side paths are intentionally rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_ALLOWED_FORMAL_RANKS = frozenset((16, 32))


@dataclass(frozen=True)
class ProjectionStats:
    """Matched dense-vs-low-rank parameter and MAC comparison."""

    in_features: int
    out_features: int
    rank: int
    bias: bool
    dense_parameters: int
    low_rank_parameters: int
    dense_macs: int
    low_rank_macs: int

    @property
    def parameter_reduction(self) -> int:
        return self.dense_parameters - self.low_rank_parameters

    @property
    def mac_reduction(self) -> int:
        return self.dense_macs - self.low_rank_macs

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "bias": self.bias,
            "dense_parameters": self.dense_parameters,
            "low_rank_parameters": self.low_rank_parameters,
            "dense_macs": self.dense_macs,
            "low_rank_macs": self.low_rank_macs,
            "parameter_reduction": self.parameter_reduction,
            "mac_reduction": self.mac_reduction,
            "parameter_ratio": self.low_rank_parameters / self.dense_parameters,
            "mac_ratio": self.low_rank_macs / self.dense_macs,
        }


@dataclass(frozen=True)
class LowRankProvenance:
    """Records how a low-rank projection was initialised."""

    init_mode: str
    rank: int
    source_name: Optional[str]
    source_shape: Optional[tuple[int, int]]
    allow_test_rank: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "init_mode": self.init_mode,
            "rank": self.rank,
            "source_name": self.source_name,
            "source_shape": (
                list(self.source_shape) if self.source_shape is not None else None
            ),
            "allow_test_rank": self.allow_test_rank,
        }


@dataclass(frozen=True)
class ProjectionConfig:
    """Dense/low-rank projection switch used by FE-2H configs."""

    kind: str = "dense"
    rank: Optional[int] = None
    bias: bool = True
    init: str = "random"
    allow_test_rank: bool = False
    source_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind == "additive_lora":
            raise ValueError(
                "additive_lora is rejected: only 'dense' or 'low_rank' "
                "projection kinds are supported"
            )
        if self.kind not in {"dense", "low_rank"}:
            raise ValueError(
                f"unsupported projection kind {self.kind!r}; expected 'dense' "
                "or 'low_rank'"
            )
        if self.kind == "dense":
            if self.rank is not None:
                raise ValueError("dense projections must not declare a rank")
            if self.init != "random":
                raise ValueError("dense projections do not support low-rank init")
            if self.source_name is not None:
                raise ValueError("dense projections must not declare source_name")
            return
        if self.rank is None:
            raise ValueError("low_rank projections require an explicit rank")
        if self.init not in {"random", "svd"}:
            raise ValueError(
                f"unsupported low-rank init {self.init!r}; expected 'random' or 'svd'"
            )
        if self.init == "svd" and not self.source_name:
            raise ValueError("svd low-rank projections require a source_name")


def dense_projection_stats(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> dict[str, int]:
    """Return dense projection parameter and per-token MAC counts."""

    dense_parameters = out_features * in_features + (out_features if bias else 0)
    dense_macs = out_features * in_features
    return {
        "dense_parameters": dense_parameters,
        "dense_macs": dense_macs,
    }


def matched_projection_report(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    bias: bool = True,
    allow_test_rank: bool = False,
) -> ProjectionStats:
    """Return a fail-closed dense-vs-low-rank report for one projection."""

    _validate_low_rank_spec(
        in_features,
        out_features,
        rank,
        allow_test_rank=allow_test_rank,
    )
    dense = dense_projection_stats(in_features, out_features, bias=bias)
    low_rank_parameters = out_features * rank + rank * in_features
    if bias:
        low_rank_parameters += out_features
    low_rank_macs = rank * (in_features + out_features)
    stats = ProjectionStats(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=bias,
        dense_parameters=dense["dense_parameters"],
        low_rank_parameters=low_rank_parameters,
        dense_macs=dense["dense_macs"],
        low_rank_macs=low_rank_macs,
    )
    if stats.low_rank_parameters >= stats.dense_parameters:
        raise ValueError(
            "low-rank projection must use strictly fewer parameters than "
            "the matched dense projection"
        )
    if stats.low_rank_macs >= stats.dense_macs:
        raise ValueError(
            "low-rank projection must use strictly fewer projection MACs than "
            "the matched dense projection"
        )
    return stats


def build_projection(
    in_features: int,
    out_features: int,
    *,
    config: ProjectionConfig,
    dense_source: Optional[nn.Linear] = None,
) -> nn.Module:
    """Build a dense or true low-rank projection from FE-2H config."""

    if config.kind == "dense":
        return nn.Linear(in_features, out_features, bias=config.bias)
    if config.rank is None:
        raise ValueError("low_rank projection config requires rank")
    if config.init == "svd":
        if dense_source is None:
            raise ValueError("svd low-rank projection requires a dense_source")
        if not isinstance(dense_source, nn.Linear):
            raise TypeError("dense_source must be an nn.Linear for svd init")
        if dense_source.in_features != in_features or dense_source.out_features != out_features:
            raise ValueError(
                "dense_source shape does not match requested projection dimensions"
            )
        if (dense_source.bias is not None) != config.bias:
            raise ValueError(
                "dense_source bias configuration must match ProjectionConfig.bias"
            )
        return LowRankLinear.from_dense(
            dense_source,
            rank=config.rank,
            source_name=config.source_name or "",
            allow_test_rank=config.allow_test_rank,
        )
    return LowRankLinear(
        in_features,
        out_features,
        rank=config.rank,
        bias=config.bias,
        allow_test_rank=config.allow_test_rank,
    )


class LowRankLinear(nn.Module):
    """Factorised linear projection that replaces a dense matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = True,
        allow_test_rank: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.allow_test_rank = bool(allow_test_rank)
        self._stats = matched_projection_report(
            self.in_features,
            self.out_features,
            self.rank,
            bias=bias,
            allow_test_rank=self.allow_test_rank,
        )
        self.left_factor = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.right_factor = nn.Parameter(torch.empty(self.rank, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self._provenance = LowRankProvenance(
            init_mode="random",
            rank=self.rank,
            source_name=None,
            source_shape=None,
            allow_test_rank=self.allow_test_rank,
        )

    def reset_parameters(self) -> None:
        right_bound = 1.0 / math.sqrt(self.in_features)
        left_bound = 1.0 / math.sqrt(self.rank)
        nn.init.uniform_(self.right_factor, -right_bound, right_bound)
        nn.init.uniform_(self.left_factor, -left_bound, left_bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -right_bound, right_bound)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.linear(inputs, self.right_factor, None)
        return F.linear(hidden, self.left_factor, self.bias)

    def equivalent_weight(self) -> Tensor:
        return self.left_factor @ self.right_factor

    def cost_report(self) -> ProjectionStats:
        return self._stats

    @property
    def provenance(self) -> LowRankProvenance:
        return self._provenance

    def provenance_dict(self) -> dict[str, Any]:
        return self._provenance.as_dict()

    @classmethod
    def from_dense(
        cls,
        dense: nn.Linear,
        rank: int,
        source_name: str,
        *,
        allow_test_rank: bool = False,
    ) -> "LowRankLinear":
        """Initialise a low-rank replacement from a dense layer via SVD."""

        if not isinstance(dense, nn.Linear):
            raise TypeError("dense must be an nn.Linear")
        if not source_name:
            raise ValueError("source_name is required for svd provenance")
        module = cls(
            dense.in_features,
            dense.out_features,
            rank,
            bias=dense.bias is not None,
            allow_test_rank=allow_test_rank,
        )
        module = module.to(device=dense.weight.device, dtype=dense.weight.dtype)
        weight = dense.weight.detach()
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]
            sigma_root = torch.sqrt(s_r)
            module.left_factor.copy_(u_r * sigma_root.unsqueeze(0))
            module.right_factor.copy_(sigma_root.unsqueeze(1) * vh_r)
            if dense.bias is not None and module.bias is not None:
                module.bias.copy_(dense.bias.detach())
        module._provenance = LowRankProvenance(
            init_mode="svd",
            rank=int(rank),
            source_name=source_name,
            source_shape=tuple(weight.shape),
            allow_test_rank=bool(allow_test_rank),
        )
        return module

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}, "
            f"allow_test_rank={self.allow_test_rank}"
        )


def _validate_low_rank_spec(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    allow_test_rank: bool,
) -> None:
    for name, value in (
        ("in_features", in_features),
        ("out_features", out_features),
        ("rank", rank),
    ):
        if int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if rank >= min(in_features, out_features):
        raise ValueError("rank must be strictly smaller than min(in_features, out_features)")
    if not allow_test_rank and rank not in _ALLOWED_FORMAL_RANKS:
        allowed = ", ".join(str(item) for item in sorted(_ALLOWED_FORMAL_RANKS))
        raise ValueError(
            f"rank must be one of {{{allowed}}} unless allow_test_rank=True"
        )


__all__ = [
    "LowRankLinear",
    "LowRankProvenance",
    "ProjectionConfig",
    "ProjectionStats",
    "build_projection",
    "dense_projection_stats",
    "matched_projection_report",
]

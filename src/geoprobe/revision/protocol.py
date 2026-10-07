"""Split isolation and manifest checks for the conditional-accept revision."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


def _normalize_ids(ids: Iterable[int], name: str) -> tuple[int, ...]:
    out = tuple(int(x) for x in ids)
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate problem IDs")
    if any(x < 0 for x in out):
        raise ValueError(f"{name} contains a negative problem ID")
    return out


@dataclass(frozen=True)
class RevisionSplit:
    """Disjoint source-training, validation and locked-test problem IDs."""

    source_train: tuple[int, ...]
    validation: tuple[int, ...]
    locked_test: tuple[int, ...]
    replication_reserve: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        groups = {
            "source_train": _normalize_ids(self.source_train, "source_train"),
            "validation": _normalize_ids(self.validation, "validation"),
            "locked_test": _normalize_ids(self.locked_test, "locked_test"),
            "replication_reserve": _normalize_ids(
                self.replication_reserve, "replication_reserve"
            ),
        }
        for name, ids in groups.items():
            if not ids:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, ids)
        names = list(groups)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = set(groups[left_name]) & set(groups[right_name])
                if overlap:
                    preview = sorted(overlap)[:10]
                    raise ValueError(
                        f"{left_name} and {right_name} overlap on problem IDs {preview}"
                    )

    @classmethod
    def gsm8k_formal(cls) -> RevisionSplit:
        """The preregistered formal split for the GSM8K steering comparison."""
        return cls(
            source_train=tuple(range(0, 100)),
            validation=tuple(range(100, 200)),
            locked_test=tuple(range(200, 300)),
            replication_reserve=tuple(range(300, 500)),
        )

    def role_for(self, problem_id: int) -> str:
        for role in ("source_train", "validation", "locked_test", "replication_reserve"):
            if problem_id in getattr(self, role):
                return role
        raise KeyError(f"problem ID {problem_id} is outside this revision split")

    def assert_direction_ids(self, ids: Iterable[int]) -> None:
        """Reject a direction that was fit using validation or locked-test IDs."""
        seen = set(_normalize_ids(ids, "direction IDs"))
        forbidden = seen & (set(self.validation) | set(self.locked_test) | set(self.replication_reserve))
        if forbidden:
            raise ValueError(
                "direction construction leaked non-training problem IDs: "
                f"{sorted(forbidden)[:10]}"
            )
        if not seen:
            raise ValueError("direction construction used no IDs")

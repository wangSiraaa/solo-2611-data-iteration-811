"""Histogram contributions of synthetic subjects and their replacement bounds.

This module turns the corrected means produced by :mod:`plate_effects.warehouse`
into the *contribution product* consumed by the brief publisher.

Model
-----
A public **roster** pre-registers subjects.  Every subject owns

* a public, pre-agreed **closed rational interval** ``[low, high]`` for its mean,
* an optional binding to at most one sample of the run (``sample_id``); a
  subject with ``sample_id is None`` is *explicitly absent*, and
* for every column, a list of **left-closed, right-open** intervals
  ``[a, b)`` with a rational weight.

A subject's contribution to a column is the sum of the weights of the
intervals that contain its mean.  An interval with ``a >= b`` is empty and
never contributes.  Absent subjects contribute the zero vector.

Replacement bound
-----------------
Only the *public agreement* (interval + rules) decides the bound; the run's
current mean does not.  For one subject the set of legal contribution vectors
is::

    { contribution(m) : m in [low, high] }  union  { zero vector (absent) }

The **replacement bound** is the maximum ``L1`` distance between any two legal
vectors::

    R = max_{x, y in S} ||x - y||_1

Because the zero vector is always legal this covers replacement by absence.
Taking a maximum over *pairs of whole vectors* (rather than summing per-column
maxima) is essential: three mutually exclusive unit intervals light one column
each, so any replacement changes at most two columns and ``R = 2`` even though
the per-column maxima add up to ``3``.  Negative weights work the same way
through ``abs`` in the ``L1`` norm.

The set of legal vectors is finite and is enumerated exactly: the contribution
function is constant between consecutive interval endpoints and may take a
different value *at* a closed left endpoint, so we evaluate every endpoint that
lies in ``[low, high]`` and one rational point in each open gap between them.
All arithmetic is exact over :class:`fractions.Fraction`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .rational import RationalError, fmt_rational, parse_rational
from .warehouse import WarehouseError


class ContributionError(ValueError):
    """Raised when a roster / rules / run combination cannot be released."""


# ---------------------------------------------------------------------------
# Columns and interval rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalRule:
    """One left-closed, right-open ``[low, high)`` interval with a weight.

    ``low >= high`` denotes an empty interval: it can never contain a mean and
    therefore never contributes.
    """

    low: Fraction
    high: Fraction
    weight: Fraction

    @property
    def is_empty(self) -> bool:
        return self.low >= self.high

    def contains(self, mean: Fraction) -> bool:
        return self.low <= mean < self.high

    @classmethod
    def parse(cls, low: object, high: object, weight: object, *, field: str = "interval") -> "IntervalRule":
        try:
            lo = parse_rational(low, field=f"{field}.low")
            hi = parse_rational(high, field=f"{field}.high")
            w = parse_rational(weight, field=f"{field}.weight")
        except RationalError as exc:
            raise ContributionError(str(exc)) from exc
        return cls(lo, hi, w)

    def to_semantic(self) -> list[str]:
        return [fmt_rational(self.low), fmt_rational(self.high), fmt_rational(self.weight)]

    def to_dict(self) -> dict:
        return {
            "low": {"exact": fmt_rational(self.low), "float": float(self.low)},
            "high": {"exact": fmt_rational(self.high), "float": float(self.high)},
            "weight": {"exact": fmt_rational(self.weight), "float": float(self.weight)},
            "empty": self.is_empty,
        }


def _column_contribution(rules: Sequence[IntervalRule], mean: Fraction) -> Fraction:
    return sum((rule.weight for rule in rules if rule.contains(mean)), Fraction(0))


# ---------------------------------------------------------------------------
# Subject declarations and the public spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectDeclaration:
    """One pre-registered subject and its public contribution agreement."""

    name: str
    low: Fraction
    high: Fraction
    sample_id: str | None
    rules: tuple[tuple[IntervalRule, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContributionError("subject name must be a non-empty string")
        if self.low > self.high:
            raise ContributionError(
                f"subject {self.name!r}: public interval [{self.low}, {self.high}] is empty"
            )
        if self.sample_id is not None:
            if not isinstance(self.sample_id, str) or not self.sample_id.strip():
                raise ContributionError(
                    f"subject {self.name!r}: sample_id must be None or a non-empty string"
                )

    def contribution(self, mean: Fraction) -> tuple[Fraction, ...]:
        return tuple(_column_contribution(column, mean) for column in self.rules)

    def to_semantic(self, n_columns: int) -> dict:
        if len(self.rules) != n_columns:
            raise ContributionError(
                f"subject {self.name!r}: {len(self.rules)} column rule sets for {n_columns} columns"
            )
        return {
            "name": self.name,
            "low": fmt_rational(self.low),
            "high": fmt_rational(self.high),
            "sample_id": self.sample_id,
            "rules": [[rule.to_semantic() for rule in column] for column in self.rules],
        }


@dataclass(frozen=True)
class HistogramSpec:
    """The public histogram agreement: ordered columns and the subject roster."""

    columns: tuple[str, ...]
    subjects: tuple[SubjectDeclaration, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ContributionError("a histogram spec needs at least one column")
        if len(set(self.columns)) != len(self.columns):
            raise ContributionError("column names must be unique")
        names: set[str] = set()
        for subject in self.subjects:
            if subject.name in names:
                raise ContributionError(f"duplicate subject name {subject.name!r}")
            names.add(subject.name)
            if len(subject.rules) != len(self.columns):
                raise ContributionError(
                    f"subject {subject.name!r}: {len(subject.rules)} rule sets for "
                    f"{len(self.columns)} columns"
                )

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def subject(self, name: str) -> SubjectDeclaration:
        for subject in self.subjects:
            if subject.name == name:
                return subject
        raise KeyError(name)

    def digest(self) -> str:
        payload = {
            "format": "histogram-spec/v1",
            "columns": list(self.columns),
            "subjects": [subject.to_semantic(len(self.columns)) for subject in self.subjects],
        }
        return _sha256(payload)

    def to_dict(self) -> dict:
        return {
            "format": "histogram-spec/v1",
            "digest": self.digest(),
            "columns": list(self.columns),
            "subjects": [
                {
                    "name": subject.name,
                    "interval": {
                        "low": {"exact": fmt_rational(subject.low), "float": float(subject.low)},
                        "high": {"exact": fmt_rational(subject.high), "float": float(subject.high)},
                    },
                    "sample_id": subject.sample_id,
                    "columns": {
                        column: [rule.to_dict() for rule in subject.rules[index]]
                        for index, column in enumerate(self.columns)
                    },
                }
                for subject in self.subjects
            ],
        }


def _sha256(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replacement bound with witnesses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundState:
    """A legal state attaining one side of a replacement bound.

    ``kind`` is ``"absent"`` or ``"mean"``.  ``mean`` is the exact realized mean
    for a ``"mean"`` state and ``None`` for absence.  ``contribution`` is the
    resulting column vector.
    """

    kind: str
    mean: Fraction | None
    contribution: tuple[Fraction, ...]

    def to_internal_dict(self) -> dict:
        return {
            "kind": self.kind,
            "mean": None if self.mean is None else fmt_rational(self.mean),
            "contribution": [fmt_rational(value) for value in self.contribution],
        }


@dataclass(frozen=True)
class ReplacementBound:
    bound: Fraction
    witnesses: tuple[BoundState, BoundState]
    legal_states: tuple[BoundState, ...]

    def to_internal_dict(self) -> dict:
        return {
            "bound": fmt_rational(self.bound),
            "witnesses": [state.to_internal_dict() for state in self.witnesses],
            "distinct_legal_vectors": len(self.legal_states),
        }


def _candidate_points(declaration: SubjectDeclaration) -> list[Fraction]:
    """Exact rational points that cover every distinct contribution value.

    Consecutive interval endpoints split ``[low, high]`` into open gaps on which
    the contribution is constant; the closed left endpoints themselves may take
    a different value from the gap on their right, so both are sampled.
    """

    low, high = declaration.low, declaration.high
    breakpoints: set[Fraction] = {low, high}
    for column in declaration.rules:
        for rule in column:
            for endpoint in (rule.low, rule.high):
                if low <= endpoint <= high:
                    breakpoints.add(endpoint)
    ordered = sorted(breakpoints)
    points: list[Fraction] = list(ordered)
    for left, right in zip(ordered, ordered[1:]):
        if left < right:
            points.append((left + right) / 2)
    # Deduplicate while preserving sorted order.
    seen: set[Fraction] = set()
    unique: list[Fraction] = []
    for point in points:
        if point not in seen:
            seen.add(point)
            unique.append(point)
    return unique


def _legal_states(declaration: SubjectDeclaration) -> tuple[BoundState, ...]:
    n_columns = len(declaration.rules)
    zero = tuple(Fraction(0) for _ in range(n_columns))
    states: list[BoundState] = [BoundState("absent", None, zero)]
    seen: set[tuple[Fraction, ...]] = {zero}
    for mean in _candidate_points(declaration):
        vector = declaration.contribution(mean)
        if vector not in seen:
            seen.add(vector)
            states.append(BoundState("mean", mean, vector))
    return tuple(states)


def replacement_bound(declaration: SubjectDeclaration) -> ReplacementBound:
    """Exact replacement bound and two legal states attaining it."""

    states = _legal_states(declaration)
    best = Fraction(0)
    best_pair = (states[0], states[0])
    for i in range(len(states)):
        left = states[i].contribution
        for j in range(i, len(states)):
            right = states[j].contribution
            distance = sum((abs(a - b) for a, b in zip(left, right)), Fraction(0))
            if distance > best:
                best = distance
                best_pair = (states[i], states[j])
    return ReplacementBound(bound=best, witnesses=best_pair, legal_states=states)


# ---------------------------------------------------------------------------
# The contribution product handed to the publisher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRef:
    """Identity of the run / layout version a product was computed from."""

    run_id: str
    layout_digest: str
    layout_version: int

    def to_semantic(self) -> dict:
        return {
            "run_id": self.run_id,
            "layout_digest": self.layout_digest,
            "layout_version": self.layout_version,
        }


@dataclass(frozen=True)
class SubjectState:
    name: str
    present: bool
    sample_id: str | None
    mean: Fraction | None
    contributions: tuple[Fraction, ...]
    replacement_bound: Fraction
    bound_witnesses: tuple[BoundState, BoundState]
    legal_states: tuple[BoundState, ...]

    def to_internal_dict(self) -> dict:
        return {
            "name": self.name,
            "present": self.present,
            "sample_id": self.sample_id,
            "mean": None if self.mean is None else fmt_rational(self.mean),
            "contributions": [fmt_rational(value) for value in self.contributions],
            "replacement_bound": fmt_rational(self.replacement_bound),
            "bound_witnesses": [state.to_internal_dict() for state in self.bound_witnesses],
            "distinct_legal_vectors": len(self.legal_states),
        }


@dataclass(frozen=True)
class ContributionProduct:
    """Everything the publisher needs: true column sums and per-subject bounds.

    ``column_sums`` are the current exact true totals.  They are *internal*:
    the publisher adds noise and never exposes them in the external brief.
    """

    spec: HistogramSpec
    source: SourceRef
    column_sums: tuple[Fraction, ...]
    subject_states: tuple[SubjectState, ...]
    digest: str

    def subject(self, name: str) -> SubjectState:
        for state in self.subject_states:
            if state.name == name:
                return state
        raise KeyError(name)

    def to_internal_dict(self) -> dict:
        return {
            "digest": self.digest,
            "spec": self.spec.to_dict(),
            "source": self.source.to_semantic(),
            "true_column_sums": {
                column: {"exact": fmt_rational(value), "float": float(value)}
                for column, value in zip(self.spec.columns, self.column_sums)
            },
            "subjects": [state.to_internal_dict() for state in self.subject_states],
        }


def _product_digest(
    spec: HistogramSpec,
    source: SourceRef,
    column_sums: Sequence[Fraction],
    subject_states: Sequence[SubjectState],
) -> str:
    payload = {
        "format": "contribution-product/v1",
        "spec": spec.digest(),
        "source": source.to_semantic(),
        "column_sums": [fmt_rational(value) for value in column_sums],
        "subjects": [
            {
                "name": state.name,
                "present": state.present,
                "sample_id": state.sample_id,
                "mean": None if state.mean is None else fmt_rational(state.mean),
                "contributions": [fmt_rational(value) for value in state.contributions],
                "replacement_bound": fmt_rational(state.replacement_bound),
            }
            for state in subject_states
        ],
    }
    return _sha256(payload)


def build_contribution_product(
    warehouse: object,
    run_or_id: object,
    spec: HistogramSpec,
) -> ContributionProduct:
    """Compute the contribution product for ``spec`` from a warehouse run.

    ``warehouse`` must expose ``run`` / ``fit`` like
    :class:`plate_effects.warehouse.ReadoutWarehouse`.  A subject that is
    declared present (``sample_id`` not ``None``) must have *complete*
    corrected evidence, and that mean must lie in the subject's public
    interval; otherwise :class:`ContributionError` is raised.  An explicitly
    absent subject (``sample_id is None``) contributes the zero vector and is
    never looked up.
    """

    if not isinstance(spec, HistogramSpec):
        raise ContributionError("spec must be a HistogramSpec")
    run = warehouse.run(run_or_id) if isinstance(run_or_id, str) else run_or_id
    report = warehouse.fit(run)
    source = SourceRef(
        run_id=report.run_id,
        layout_digest=report.layout_digest,
        layout_version=report.layout_version,
    )
    known_samples = {summary.sample_id for summary in report.samples}

    n_columns = spec.n_columns
    column_sums = [Fraction(0) for _ in range(n_columns)]
    states: list[SubjectState] = []

    for declaration in spec.subjects:
        bound = replacement_bound(declaration)
        if declaration.sample_id is None:
            present = False
            mean: Fraction | None = None
            contributions = tuple(Fraction(0) for _ in range(n_columns))
        else:
            if declaration.sample_id not in known_samples:
                raise ContributionError(
                    f"subject {declaration.name!r}: sample {declaration.sample_id!r} "
                    "is not part of the run (no evidence)"
                )
            # The single corrected mean comes from the warehouse entry point,
            # which raises for a missing or unidentifiable replicate.
            try:
                mean = warehouse.complete_sample_mean(run, declaration.sample_id)
            except WarehouseError as exc:
                raise ContributionError(
                    f"subject {declaration.name!r}: sample {declaration.sample_id!r} "
                    f"has no complete evidence ({exc})"
                ) from exc
            if not (declaration.low <= mean <= declaration.high):
                raise ContributionError(
                    f"subject {declaration.name!r}: mean {fmt_rational(mean)} is outside the "
                    f"agreed closed interval [{fmt_rational(declaration.low)}, "
                    f"{fmt_rational(declaration.high)}]"
                )
            present = True
            contributions = declaration.contribution(mean)

        for index, value in enumerate(contributions):
            column_sums[index] += value

        states.append(
            SubjectState(
                name=declaration.name,
                present=present,
                sample_id=declaration.sample_id,
                mean=mean,
                contributions=contributions,
                replacement_bound=bound.bound,
                bound_witnesses=bound.witnesses,
                legal_states=bound.legal_states,
            )
        )

    digest = _product_digest(spec, source, column_sums, states)
    return ContributionProduct(
        spec=spec,
        source=source,
        column_sums=tuple(column_sums),
        subject_states=tuple(states),
        digest=digest,
    )

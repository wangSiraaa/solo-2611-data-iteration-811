"""Bundled synthetic scenarios for histogram contributions and brief publishing.

Everything here is synthetic.  The scenarios exist so the contribution bounds,
the budget accounting, the noise injection and the idempotent re-read can all be
replayed and audited independently.

The central example is the one from the task statement:

* subject ``u`` has legal mean interval ``[0, 2]`` and two columns.  Three legal
  states give contribution vectors ``(0, 0)``, ``(2, 0)`` and ``(0, 3)``; the
  last two are produced by the intervals ``[0, 1)`` and ``[1, 3)`` that each
  drive one unit column.  The replacement bound is ``5`` (``(2,0)`` vs
  ``(0,3)`` in ``L1``).
* subject ``v`` has legal vectors ``(0, 0)`` and ``(1, 0)``, bound ``1``.

At scale ``2`` with budgets ``2`` and ``1`` the brief is refused.  Narrowing
``u``'s public interval to ``[1, 2]`` (mean still valid) lowers its bound to
``3``, and ``3/2 <= 2`` (the ``u`` budget) so the same budget now publishes.  If
instead ``u``'s budget is raised to ``3``, the original brief is accepted and the
budgets fall to ``u = 1/2`` and ``v = 1/2``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from .contributions import (
    HistogramSpec,
    IntervalRule,
    SubjectDeclaration,
    build_contribution_product,
)
from .design import PlateDesign, PublishedLayout, Well
from .publisher import BriefPublisher, PublicationRejected
from .warehouse import ReadoutWarehouse


F = Fraction


@dataclass
class ExampleResult:
    name: str
    description: str
    payload: Mapping[str, object]


def _rule(low: object, high: object, weight: object) -> IntervalRule:
    return IntervalRule(F(low), F(high), F(weight))


def u_v_spec(*, narrow_u: bool = False) -> HistogramSpec:
    """The two-column ``u``/``v`` agreement from the task statement."""

    u_low, u_high = (F(1), F(2)) if narrow_u else (F(0), F(2))
    u = SubjectDeclaration(
        name="u",
        low=u_low,
        high=u_high,
        sample_id="S-u",
        rules=(
            (_rule(0, 1, 2),),  # column 0 gets weight 2 for means in [0, 1)
            (_rule(1, 3, 3),),  # column 1 gets weight 3 for means in [1, 3)
        ),
    )
    v = SubjectDeclaration(
        name="v",
        low=F(0),
        high=F(1),
        sample_id="S-v",
        rules=(
            (_rule(0, 1, 1),),  # column 0 lit for means in [0, 1)
            (),
        ),
    )
    return HistogramSpec(columns=("low", "high"), subjects=(u, v))


def _u_v_layout_and_run() -> tuple[ReadoutWarehouse, str, PublishedLayout]:
    """A plate whose S-u and S-v samples are exactly recoverable.

    ``S-u`` has corrected mean ``1`` (so ``u`` lands in ``[1, 3)`` -> ``(0, 3)``)
    and ``S-v`` has corrected mean ``1/2`` (so ``v`` -> ``(1, 0)``).  That is the
    state whose bound witnesses the task describes.
    """

    design = PlateDesign("plate-uv", rows=8, cols=12, min_replicates=1, max_replicates=2)
    for row in range(8):
        for col in range(12):
            name = Well(row, col).name
            if (row + col) % 2 == 0:
                design.set_blank([name])
            else:
                design.set_reference([name], F(row + 1, 2))
    design.define_sample("S-u", ["A1"])
    design.define_sample("S-v", ["B2"])
    layout = design.publish()

    b = F(1, 3)
    row_effects = {i: F(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: F(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-u": F(1), "S-v": F(1, 2)}
    measurements = {}
    for well in layout.wells:
        if well.role == "unused":
            continue
        bias = b + row_effects[well.row] + col_effects[well.col]
        if well.role == "blank":
            base = F(0)
        elif well.role == "reference":
            base = well.expected
        else:
            base = sample_values[well.sample_id]
        measurements[well.name] = base + bias

    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="uv-run")
    return warehouse, run.run_id, layout


def build_uv_bounds_example() -> ExampleResult:
    """Bounds and witnesses for the ``u``/``v`` example, no budgets involved."""

    warehouse, run_id, _ = _u_v_layout_and_run()
    product = build_contribution_product(warehouse, run_id, u_v_spec())
    narrow_product = build_contribution_product(warehouse, run_id, u_v_spec(narrow_u=True))
    return ExampleResult(
        name="uv_bounds",
        description=(
            "u's bound is 5 (witnesses (2,0) and (0,3)); v's bound is 1.  Narrowing "
            "u's public interval to [1,2] leaves the mean 1 valid and lowers the bound "
            "to 3."
        ),
        payload={
            "true_column_sums": [str(v) for v in product.column_sums],
            "u": product.subject("u").to_internal_dict(),
            "v": product.subject("v").to_internal_dict(),
            "u_narrowed_interval_bound": str(
                narrow_product.subject("u").replacement_bound
            ),
        },
    )


def build_uv_budget_example() -> ExampleResult:
    """The three budget outcomes from the task statement."""

    warehouse, run_id, _ = _u_v_layout_and_run()
    product = build_contribution_product(warehouse, run_id, u_v_spec())
    narrow_product = build_contribution_product(warehouse, run_id, u_v_spec(narrow_u=True))

    # 1. scale 2, budgets 2 and 1 -> refused entirely.
    refused_publisher = BriefPublisher({"u": 2, "v": 1})
    refused = False
    try:
        refused_publisher.publish("uv-1", product, scale=2)
    except PublicationRejected as exc:
        refused = True
        refusal_detail = {
            "shortfalls": {k: dict(v) for k, v in exc.shortfalls.items()}
        }
    remaining_after_refusal = {
        name: str(value) for name, value in refused_publisher.ledger.remaining_all().items()
    }

    # 2. Same budgets, narrowed u interval -> bound 3 -> 3/2 <= 2, publishable.
    narrowed_publisher = BriefPublisher({"u": 2, "v": 1})
    narrowed_brief = narrowed_publisher.publish("uv-2", narrow_product, scale=2)
    narrowed_remaining = {
        name: str(value)
        for name, value in narrowed_publisher.ledger.remaining_all().items()
    }

    # 3. Original product, u budget raised to 3 -> accepted; remaining u = 1/2,
    #    v = 1/2.
    accepted_publisher = BriefPublisher({"u": 3, "v": 1})
    accepted_brief = accepted_publisher.publish("uv-3", product, scale=2)
    accepted_remaining = {
        name: str(value)
        for name, value in accepted_publisher.ledger.remaining_all().items()
    }

    return ExampleResult(
        name="uv_budgets",
        description=(
            "Scale 2 with budgets (2,1) refuses the whole brief.  Narrowing u to "
            "[1,2] makes it publishable on the same budget.  Raising u's budget to 3 "
            "accepts the original brief and leaves (1/2, 1/2)."
        ),
        payload={
            "refused": refused,
            "refusal": refusal_detail,
            "remaining_after_refusal": remaining_after_refusal,
            "narrowed_published": narrowed_brief["publication_id"],
            "narrowed_remaining": narrowed_remaining,
            "accepted_published": accepted_brief["publication_id"],
            "accepted_remaining": accepted_remaining,
        },
    )


# ---------------------------------------------------------------------------
# A richer multi-subject histogram with empty intervals and negative weights
# ---------------------------------------------------------------------------


def _histogram_layout_and_run() -> tuple[ReadoutWarehouse, str]:
    design = PlateDesign("plate-hist", rows=8, cols=12, min_replicates=1, max_replicates=2)
    for row in range(8):
        for col in range(12):
            name = Well(row, col).name
            if (row + col) % 2 == 0:
                design.set_blank([name])
            else:
                design.set_reference([name], F(row + 1, 2))
    for sample, well in (("S-a", "A1"), ("S-b", "B2"), ("S-c", "C3")):
        design.define_sample(sample, [well])
    layout = design.publish()

    b = F(1, 5)
    row_effects = {i: F(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: F(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-a": F(1, 2), "S-b": F(3, 2), "S-c": F(5, 2)}
    measurements = {}
    for well in layout.wells:
        if well.role == "unused":
            continue
        bias = b + row_effects[well.row] + col_effects[well.col]
        if well.role == "blank":
            base = F(0)
        elif well.role == "reference":
            base = well.expected
        else:
            base = sample_values[well.sample_id]
        measurements[well.name] = base + bias
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="hist-run")
    return warehouse, run.run_id


def histogram_spec() -> HistogramSpec:
    """Three columns, two present subjects, one absent, plus edge cases."""

    a = SubjectDeclaration(
        name="a",
        low=F(0),
        high=F(2),
        sample_id="S-a",
        rules=(
            (_rule(0, 1, 1), _rule(1, 2, 2)),   # col0: staircase
            (_rule(1, 1, 5),),                   # col1: empty interval, never lit
            (_rule(0, 1, -1), _rule(1, 2, -2)),  # col2: negative weights
        ),
    )
    b = SubjectDeclaration(
        name="b",
        low=F(0),
        high=F(3),
        sample_id="S-b",
        rules=(
            (_rule(0, 1, 1),),                   # col0 unit on [0,1)
            (_rule(1, 2, 1),),                   # col1 unit on [1,2)
            (_rule(2, 3, 1),),                   # col2 unit on [2,3)
        ),
    )
    absent = SubjectDeclaration(
        name="c-absent",
        low=F(-1),
        high=F(1),
        sample_id=None,
        rules=(
            (_rule(-1, 0, 2), _rule(0, 1, 4)),
            (_rule(-1, 1, -3),),
            (),
        ),
    )
    return HistogramSpec(
        columns=("zero", "one", "two"), subjects=(a, b, absent)
    )


def build_histogram_example() -> ExampleResult:
    warehouse, run_id = _histogram_layout_and_run()
    product = build_contribution_product(warehouse, run_id, histogram_spec())
    return ExampleResult(
        name="histogram_bounds",
        description=(
            "Three columns with a staircase, an empty interval, negative weights and "
            "an explicitly absent subject.  Mutually exclusive unit columns make b's "
            "bound 2, not the 3 a per-column sum would give."
        ),
        payload={
            "true_column_sums": [str(v) for v in product.column_sums],
            "subject_states": {
                state.name: state.to_internal_dict() for state in product.subject_states
            },
        },
    )

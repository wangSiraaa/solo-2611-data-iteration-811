"""Tests for histogram contributions, replacement bounds and the publisher.

Every case uses synthetic data.  Noise is injected deterministically so the
ledger accounting and the noisy output can each be checked independently, and
the ``u``/``v`` example from the task statement is asserted verbatim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from fractions import Fraction

import pytest

from plate_effects.contributions import (
    ContributionError,
    HistogramSpec,
    IntervalRule,
    SubjectDeclaration,
    build_contribution_product,
    replacement_bound,
)
from plate_effects.histogram_scenarios import (
    _histogram_layout_and_run,
    _u_v_layout_and_run,
    build_histogram_example,
    build_uv_bounds_example,
    build_uv_budget_example,
    histogram_spec,
    u_v_spec,
)
from plate_effects.publisher import (
    BriefPublisher,
    BudgetError,
    PublicationConflict,
    PublicationRejected,
    PublisherError,
)
from plate_effects.warehouse import ReadoutWarehouse


F = Fraction


def _rule(low, high, weight):
    return IntervalRule(F(low), F(high), F(weight))


def _product(*, narrow_u=False):
    warehouse, run_id, _ = _u_v_layout_and_run()
    return build_contribution_product(warehouse, run_id, u_v_spec(narrow_u=narrow_u))


def _fixed_noise(values):
    """A noise source that ignores the scale and yields fixed rationals."""

    def source(count, scale):
        assert count == len(values)
        return [F(v) for v in values]

    return source


# ---------------------------------------------------------------------------
# Interval semantics
# ---------------------------------------------------------------------------


def test_interval_half_open_and_empty():
    rule = _rule(0, 1, 5)
    assert rule.contains(F(0))          # closed left
    assert not rule.contains(F(1))      # open right
    assert rule.contains(F(1, 2))
    empty = _rule(1, 1, 7)
    assert empty.is_empty
    assert not empty.contains(F(1))
    reversed_empty = _rule(2, 1, 7)
    assert reversed_empty.is_empty


def test_contribution_accumulates_and_negative_weights():
    declaration = SubjectDeclaration(
        name="s",
        low=F(0),
        high=F(3),
        sample_id=None,
        rules=(
            (_rule(0, 2, 2), _rule(1, 3, 3), _rule(1, 1, 9)),  # empty ignored
            (_rule(0, 3, -1),),
        ),
    )
    assert declaration.contribution(F(0)) == (F(2), F(-1))     # only [0,2)
    assert declaration.contribution(F(1)) == (F(5), F(-1))     # [0,2)+[1,3)
    assert declaration.contribution(F(2)) == (F(3), F(-1))     # only [1,3)
    assert declaration.contribution(F(3)) == (F(0), F(0))      # open right, none


# ---------------------------------------------------------------------------
# Replacement bounds: the task's worked example
# ---------------------------------------------------------------------------


def test_uv_bounds_are_five_and_one():
    product = _product()
    u = product.subject("u")
    v = product.subject("v")
    assert u.replacement_bound == F(5)
    assert v.replacement_bound == F(1)
    u_vectors = {state.contribution for state in u.bound_witnesses}
    assert u_vectors == {(F(2), F(0)), (F(0), F(3))}
    v_vectors = {state.contribution for state in v.bound_witnesses}
    assert v_vectors == {(F(0), F(0)), (F(1), F(0))}


def test_uv_narrowing_lowers_u_bound_to_three():
    narrow = _product(narrow_u=True)
    assert narrow.subject("u").replacement_bound == F(3)


def test_mutually_exclusive_columns_bound_is_two_not_three():
    """The three unit columns of ``b`` can change at most two per replacement."""

    product = build_contribution_product(*_histogram_layout_and_run(), histogram_spec())
    b = product.subject("b")
    assert b.replacement_bound == F(2)
    # Per-column maxima would (wrongly) add up to 3.
    per_column = [max(abs(s.contribution[i]) for s in b.legal_states) for i in range(3)]
    assert sum(per_column) == F(3)
    assert b.replacement_bound == F(2)


def test_bound_witnesses_really_attain_the_bound():
    product = build_contribution_product(*_histogram_layout_and_run(), histogram_spec())
    for state in product.subject_states:
        left, right = state.bound_witnesses
        distance = sum(abs(a - b) for a, b in zip(left.contribution, right.contribution))
        assert distance == state.replacement_bound
        # Each witness is a legal state: absent, or a mean inside the interval.
        for witness in state.bound_witnesses:
            if witness.kind == "absent":
                assert witness.mean is None
            else:
                declaration = product.spec.subject(state.name)
                assert declaration.low <= witness.mean <= declaration.high


def test_absent_subject_still_has_its_own_bound():
    product = build_contribution_product(*_histogram_layout_and_run(), histogram_spec())
    absent = product.subject("c-absent")
    assert not absent.present
    assert absent.contributions == (F(0), F(0), F(0))
    # Bound is a property of the public agreement alone, so it is non-zero.
    assert absent.replacement_bound == F(7)


def test_bound_ignores_the_current_mean():
    """Same public agreement, different realized state -> same bound."""

    declaration = SubjectDeclaration(
        name="x", low=F(0), high=F(2), sample_id=None,
        rules=((_rule(0, 1, 1),),),
    )
    assert replacement_bound(declaration).bound == F(1)


def test_closed_endpoint_and_open_right_are_accounted_for():
    # The value at the closed left endpoint can differ from the gap to its right.
    declaration = SubjectDeclaration(
        name="edge", low=F(0), high=F(2), sample_id=None,
        rules=((_rule(0, 1, 1), _rule(0, 2, 10)),),
    )
    # At mean 0: 1 + 10 = 11.  In (0,2): 10.  Absent: 0.
    bound = replacement_bound(declaration)
    assert bound.bound == F(11)
    vectors = {state.contribution for state in bound.legal_states}
    assert vectors == {(F(11),), (F(10),), (F(0),)}


# ---------------------------------------------------------------------------
# Product building: validation of present subjects
# ---------------------------------------------------------------------------


def test_present_subject_missing_evidence_raises():
    warehouse, run_id, _ = _u_v_layout_and_run()
    spec = HistogramSpec(
        columns=("c",),
        subjects=(
            SubjectDeclaration(
                name="ghost", low=F(0), high=F(1), sample_id="S-does-not-exist",
                rules=((_rule(0, 1, 1),),),
            ),
        ),
    )
    with pytest.raises(ContributionError):
        build_contribution_product(warehouse, run_id, spec)


def test_present_subject_mean_outside_interval_raises():
    warehouse, run_id, _ = _u_v_layout_and_run()
    spec = HistogramSpec(
        columns=("c",),
        subjects=(
            # S-u's corrected mean is 1, outside [2, 3].
            SubjectDeclaration(
                name="u", low=F(2), high=F(3), sample_id="S-u",
                rules=((_rule(2, 3, 1),),),
            ),
        ),
    )
    with pytest.raises(ContributionError):
        build_contribution_product(warehouse, run_id, spec)


def test_present_subject_with_incomplete_evidence_raises():
    from plate_effects.design import PlateDesign, Well

    design = PlateDesign("plate-incomplete", rows=8, cols=12, min_replicates=2, max_replicates=2)
    for r in range(8):
        for c in range(12):
            name = Well(r, c).name
            if (r + c) % 2 == 0:
                design.set_blank([name])
            else:
                design.set_reference([name], F(r + 1, 2))
    design.define_sample("S-partial", ["A1", "B1"])
    layout = design.publish()
    measurements = {}
    for well in layout.wells:
        if well.role == "unused":
            continue
        if well.role == "blank":
            measurements[well.name] = F(0)
        elif well.role == "reference":
            measurements[well.name] = well.expected
        else:
            measurements[well.name] = F(3)
    del measurements["B1"]
    wh = ReadoutWarehouse()
    run = wh.import_run(layout, measurements)
    spec = HistogramSpec(
        columns=("c",),
        subjects=(
            SubjectDeclaration(
                name="p", low=F(0), high=F(10), sample_id="S-partial",
                rules=((_rule(0, 10, 1),),),
            ),
        ),
    )
    with pytest.raises(ContributionError):
        build_contribution_product(wh, run, spec)


def test_product_true_sums_match_contributions():
    product = _product()
    expected = [
        sum((state.contributions[i] for state in product.subject_states), F(0))
        for i in range(product.spec.n_columns)
    ]
    assert list(product.column_sums) == expected


# ---------------------------------------------------------------------------
# Warehouse entry-point extension
# ---------------------------------------------------------------------------


def test_complete_sample_mean_returns_exact_mean():
    warehouse, run_id, _ = _u_v_layout_and_run()
    assert warehouse.complete_sample_mean(run_id, "S-u") == F(1)
    assert warehouse.complete_sample_mean(run_id, "S-v") == F(1, 2)


def test_complete_sample_mean_raises_on_unknown_sample():
    from plate_effects.warehouse import WarehouseError

    warehouse, run_id, _ = _u_v_layout_and_run()
    with pytest.raises(WarehouseError):
        warehouse.complete_sample_mean(run_id, "S-nope")


def test_subject_means_handles_absence():
    warehouse, run_id, _ = _u_v_layout_and_run()
    means = warehouse.subject_means(run_id, {"u": "S-u", "missing": None})
    assert means == {"u": F(1), "missing": None}


# ---------------------------------------------------------------------------
# Publisher: budgets and refusal
# ---------------------------------------------------------------------------


def test_refusal_charges_nothing_and_reveals_nothing():
    product = _product()
    publisher = BriefPublisher({"u": 2, "v": 1}, noise_source=_fixed_noise([1, 2]))
    with pytest.raises(PublicationRejected) as info:
        publisher.publish("uv-1", product, scale=2)
    assert "u" in info.value.shortfalls
    # No charge, no publication, empty initial state intact.
    assert publisher.ledger.remaining_all() == {"u": F(2), "v": F(1)}
    with pytest.raises(KeyError):
        publisher.brief("uv-1")


def test_narrowed_interval_publishes_on_same_budget():
    product = _product(narrow_u=True)
    publisher = BriefPublisher({"u": 2, "v": 1}, noise_source=_fixed_noise([0, 0]))
    brief = publisher.publish("uv-2", product, scale=2)
    assert brief["publication_id"] == "uv-2"
    # charges: u=3/2, v=1/2
    assert publisher.ledger.remaining_all() == {"u": F(1, 2), "v": F(1, 2)}


def test_original_product_with_raised_u_budget_leaves_half_half():
    product = _product()
    publisher = BriefPublisher({"u": 3, "v": 1}, noise_source=_fixed_noise([0, 0]))
    publisher.publish("uv-3", product, scale=2)
    assert publisher.ledger.remaining_all() == {"u": F(1, 2), "v": F(1, 2)}


def test_budget_charge_is_bound_over_scale():
    product = _product()
    publisher = BriefPublisher({"u": 100, "v": 100}, noise_source=_fixed_noise([0, 0]))
    publisher.publish("p", product, scale=F(5, 2))
    # u: 5 / (5/2) = 2, v: 1 / (5/2) = 2/5.
    assert publisher.ledger.remaining_all() == {"u": F(98), "v": F(100) - F(2, 5)}


# ---------------------------------------------------------------------------
# Publisher: noise injected independently of accounting
# ---------------------------------------------------------------------------


def test_known_noise_is_added_and_auditable():
    product = _product()
    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=_fixed_noise([F(1, 3), F(-2)]))
    brief = publisher.publish("n", product, scale=1)
    # True sums for the u/v product at S-u=1, S-v=1/2: (0,3)+(1,0) = (1,3).
    assert product.column_sums == (F(1), F(3))
    recorded = publisher.internal_record("n")["noised"]
    assert recorded[0]["exact"] == "4/3"
    assert recorded[1]["exact"] == "1"
    assert [c["noised"]["exact"] for c in brief["columns"]] == ["4/3", "1"]


def test_float_noise_source_is_supported():
    product = _product()

    def source(count, scale):
        return [0.25, -0.5]

    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=source)
    brief = publisher.publish("f", product, scale=1)
    assert brief["columns"][0]["noised"]["float"] == pytest.approx(1.25)
    assert brief["columns"][1]["noised"]["float"] == pytest.approx(2.5)


def test_brief_exposes_only_public_parameters_and_noisy_columns():
    product = _product()
    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=_fixed_noise([0, 0]))
    brief = publisher.publish("t", product, scale=1)
    assert set(brief) == {
        "format",
        "publication_id",
        "spec_digest",
        "source",
        "scale",
        "replacement_bounds",
        "columns",
    }
    text = json.dumps(brief)
    # The exact true totals are not present in the external brief.
    assert "true_column_sums" not in text
    # Each noised value is published exactly and numerically.
    for column in brief["columns"]:
        assert set(column["noised"]) == {"exact", "float"}


# ---------------------------------------------------------------------------
# Publisher: idempotent re-read and conflicts
# ---------------------------------------------------------------------------


def test_reread_returns_same_values_and_charges_once():
    product = _product()
    calls = []

    def source(count, scale):
        calls.append((count, scale))
        return [F(1), F(2)]

    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=source)
    first = publisher.publish("r", product, scale=2)
    remaining_after_first = publisher.ledger.remaining_all()
    second = publisher.publish("r", product, scale=2)
    third = publisher.brief("r")
    assert first == second == third
    assert len(calls) == 1                    # noise drawn exactly once
    assert publisher.ledger.remaining_all() == remaining_after_first


def test_conflicting_input_product_is_rejected():
    product = _product()
    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=_fixed_noise([0, 0]))
    publisher.publish("x", product, scale=1)
    other = _product(narrow_u=True)  # different rule set -> different digest
    with pytest.raises(PublicationConflict):
        publisher.publish("x", other, scale=1)


def test_conflicting_scale_is_rejected():
    product = _product()
    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=_fixed_noise([0, 0]))
    publisher.publish("y", product, scale=1)
    with pytest.raises(PublicationConflict):
        publisher.publish("y", product, scale=2)


def test_conflicting_source_version_is_rejected():
    product = _product()
    publisher = BriefPublisher({"u": 10, "v": 10}, noise_source=_fixed_noise([0, 0]))
    publisher.publish("z", product, scale=1)
    # Build a product from a different run but the same spec would differ in
    # source; simulate by constructing a product whose source differs.
    from plate_effects.contributions import SourceRef, ContributionProduct

    different_source = ContributionProduct(
        spec=product.spec,
        source=SourceRef(run_id="other-run", layout_digest="deadbeef", layout_version=99),
        column_sums=product.column_sums,
        subject_states=product.subject_states,
        digest="different",
    )
    with pytest.raises(PublicationConflict):
        publisher.publish("z", different_source, scale=1)


def test_invalid_scale_and_budget_rejected():
    product = _product()
    with pytest.raises(BudgetError):
        BriefPublisher({"u": -1})
    with pytest.raises(BudgetError):
        BriefPublisher({"u": F(1)}, noise_source=_fixed_noise([0, 0])).publish(
            "s", product, scale=0
        )
    with pytest.raises(PublisherError):
        BriefPublisher({"u": 10, "v": 10}).publish("", product, scale=1)


# ---------------------------------------------------------------------------
# Publisher: durability, reload and concurrency
# ---------------------------------------------------------------------------


def test_journal_reload_preserves_ledger_state(tmp_path):
    path = tmp_path / "ledger.jsonl"
    product = _product()
    publisher = BriefPublisher(
        {"u": 3, "v": 1}, journal_path=path, noise_source=_fixed_noise([F(1), F(2)])
    )
    brief = publisher.publish("persist", product, scale=2)
    assert publisher.ledger.remaining_all() == {"u": F(1, 2), "v": F(1, 2)}

    reopened = BriefPublisher({"u": 3, "v": 1}, journal_path=path)
    assert reopened.ledger.remaining_all() == {"u": F(1, 2), "v": F(1, 2)}
    assert reopened.brief("persist") == brief
    # Re-reading after reopening still charges nothing.
    assert reopened.publish("persist", product, scale=2) == brief
    assert reopened.ledger.remaining_all() == {"u": F(1, 2), "v": F(1, 2)}


def test_audit_conserves_charges(tmp_path):
    path = tmp_path / "ledger.jsonl"
    product = _product()
    publisher = BriefPublisher(
        {"u": 3, "v": 1}, journal_path=path, noise_source=_fixed_noise([0, 0])
    )
    publisher.publish("a", product, scale=2)
    audit = publisher.ledger.audit()
    assert audit["ok"]
    assert audit["conserved"]
    assert audit["nonnegative"]
    assert audit["publications"] == 1


def test_journal_header_mismatch_rejected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    product = _product()
    BriefPublisher({"u": 3, "v": 1}, journal_path=path, noise_source=_fixed_noise([0, 0])).publish(
        "a", product, scale=2
    )
    with pytest.raises(BudgetError):
        BriefPublisher({"u": 99, "v": 1}, journal_path=path)


def test_concurrent_last_budget_serialises_across_threads(tmp_path):
    """Several same-process instances share one journal; only one wins."""

    path = tmp_path / "ledger.jsonl"
    warehouse, run_id, _ = _u_v_layout_and_run()
    product = build_contribution_product(warehouse, run_id, u_v_spec())
    outcomes = []

    def attempt(index):
        publisher = BriefPublisher(
            {"u": 1, "v": 10}, journal_path=path,
            noise_source=lambda count, scale: [F(0)] * count,
        )
        try:
            publisher.publish(f"p{index}", product, scale=5)
            outcomes.append("ACCEPTED")
        except PublicationRejected:
            outcomes.append("REJECTED")

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("ACCEPTED") == 1
    assert outcomes.count("REJECTED") == 5
    reopened = BriefPublisher({"u": 1, "v": 10}, journal_path=path)
    assert reopened.ledger.audit()["ok"]
    assert reopened.ledger.remaining_all()["u"] == F(0)


def test_concurrent_last_budget_serialises_across_processes(tmp_path):
    """Two processes race for the last unit; exactly one wins, ledger reopens clean."""

    path = tmp_path / "ledger.jsonl"
    # Budget chosen so exactly one of two charges fits: each charge is
    # bound/scale = 5/5 = 1 for u; but v charges 1/5 too, so start v with plenty.
    # u budget 1, two competing publications each charge 1 -> one succeeds.
    product = _product()
    # Persist the product via a tiny driver so both processes see identical input.
    product_json = {
        "spec": product.spec.to_dict(),
        "source": product.source.to_semantic(),
        "column_sums": [str(v) for v in product.column_sums],
        "digest": product.digest,
    }
    product_path = tmp_path / "product.json"
    product_path.write_text(json.dumps(product_json))

    driver = textwrap.dedent(
        f"""
        import json, sys
        from fractions import Fraction
        from plate_effects.contributions import build_contribution_product
        from plate_effects.histogram_scenarios import _u_v_layout_and_run, u_v_spec
        from plate_effects.publisher import BriefPublisher, PublicationRejected

        pub_id = sys.argv[1]
        journal = sys.argv[2]
        wh, run_id, _ = _u_v_layout_and_run()
        product = build_contribution_product(wh, run_id, u_v_spec())
        pub = BriefPublisher(
            {{"u": 1, "v": 10}}, journal_path=journal,
            noise_source=lambda count, scale: [Fraction(0)] * count,
        )
        try:
            pub.publish(pub_id, product, scale=5)
            print("ACCEPTED")
        except PublicationRejected:
            print("REJECTED")
        """
    )
    driver_path = tmp_path / "driver.py"
    driver_path.write_text(driver)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))

    procs = []
    for pub_id in ("race-a", "race-b"):
        procs.append(
            subprocess.Popen(
                [sys.executable, str(driver_path), pub_id, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        )
    outcomes = []
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        outcomes.append(out.strip())

    assert sorted(outcomes) == ["ACCEPTED", "REJECTED"]

    # Reopen the ledger and verify conservation.
    reopened = BriefPublisher({"u": 1, "v": 10}, journal_path=path)
    audit = reopened.ledger.audit()
    assert audit["ok"]
    remaining = reopened.ledger.remaining_all()
    assert remaining["u"] == F(0)             # exactly one unit consumed
    assert remaining["v"] == F(10) - F(1, 5)  # one v charge of 1/5


# ---------------------------------------------------------------------------
# Scenario payloads
# ---------------------------------------------------------------------------


def test_uv_bounds_example_payload():
    payload = build_uv_bounds_example().payload
    assert payload["u"]["replacement_bound"] == "5"
    assert payload["v"]["replacement_bound"] == "1"
    assert payload["u_narrowed_interval_bound"] == "3"


def test_uv_budget_example_payload():
    payload = build_uv_budget_example().payload
    assert payload["refused"] is True
    assert payload["remaining_after_refusal"] == {"u": "2", "v": "1"}
    assert payload["narrowed_remaining"] == {"u": "1/2", "v": "1/2"}
    assert payload["accepted_remaining"] == {"u": "1/2", "v": "1/2"}


def test_histogram_example_payload():
    payload = build_histogram_example().payload
    assert payload["subject_states"]["b"]["replacement_bound"] == "2"
    assert payload["subject_states"]["c-absent"]["present"] is False
    assert payload["subject_states"]["c-absent"]["replacement_bound"] == "7"

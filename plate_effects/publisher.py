"""Brief publisher with per-subject replacement-bound budgets and Laplace noise.

The publisher accepts a :class:`~plate_effects.contributions.ContributionProduct`
(its true column totals plus one replacement bound per subject) and produces an
*external brief*: a publication id, the public parameters, and the noisy column
totals.  The exact totals, the roster and the bound witnesses stay internal.

Accounting
----------
Each subject has an independent remaining budget.  Publishing charges, for every
subject, its replacement bound divided by the publication's rational scale
``b > 0``::

    charge(subject) = replacement_bound(subject) / b

If *any* subject's budget is too small the **whole** brief is refused: no subject
is charged and no un-noised total is revealed.  A subject that is currently
absent still has (and is charged) its bound.

Idempotency
-----------
A publication id is accepted at most once.  Re-reading an accepted id returns the
very same noisy values and charges nothing further.  Re-using an id with a
different input product, a different source version, or a different scale is
rejected.

Durability and concurrency
--------------------------
When a journal path is supplied the ledger is append-only and fsync'ed under an
exclusive file lock.  Every transaction re-reads the journal, so concurrent
processes contending for the last unit of budget serialize cleanly: exactly one
wins and the others are refused.  The journal can be reopened later and audited
to confirm that ``initial - sum(charges) == remaining`` for every subject.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping, Sequence

try:  # pragma: no cover - platform dependent import
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

from .contributions import ContributionProduct
from .rational import RationalError, fmt_rational, parse_rational

LEDGER_FORMAT = "brief-ledger/v1"

# A noise source receives the number of columns and the rational scale and must
# return that many values.  Returning :class:`Fraction` keeps the arithmetic
# exact (used by tests); returning floats models a real Laplace sampler.
NoiseSource = Callable[[int, Fraction], Sequence[object]]


class PublisherError(ValueError):
    """Base class for publisher errors."""


class PublicationRejected(PublisherError):
    """Refused because at least one subject's budget is insufficient."""

    def __init__(self, publication_id: str, shortfalls: Mapping[str, dict]) -> None:
        self.publication_id = publication_id
        self.shortfalls = dict(shortfalls)
        detail = "; ".join(
            f"{name}: need {info['charge']} have {info['remaining']}"
            for name, info in self.shortfalls.items()
        )
        super().__init__(
            f"publication {publication_id!r} refused (no subject charged, no totals "
            f"revealed): {detail}"
        )


class PublicationConflict(PublisherError):
    """An accepted/used id is being re-used with different input or parameters."""


class BudgetError(PublisherError):
    """Invalid budget / scale configuration."""


# ---------------------------------------------------------------------------
# Durable ledger
# ---------------------------------------------------------------------------


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublicationRecord:
    publication_id: str
    fingerprint: str
    charges: dict[str, Fraction]
    noised: tuple[object, ...]
    scale: Fraction
    spec_digest: str
    brief: dict

    def to_journal(self) -> dict:
        return {
            "type": "publication",
            "publication_id": self.publication_id,
            "fingerprint": self.fingerprint,
            "charges": {name: fmt_rational(value) for name, value in self.charges.items()},
            "noised": [_encode_number(value) for value in self.noised],
            "scale": fmt_rational(self.scale),
            "spec_digest": self.spec_digest,
            "brief": self.brief,
        }


def _encode_number(value: object) -> dict:
    if isinstance(value, Fraction):
        return {"exact": fmt_rational(value), "float": float(value)}
    return {"float": float(value)}


class BudgetLedger:
    """Per-subject budgets with an optional append-only journal.

    The ledger is the single source of truth for charges.  All mutation happens
    inside :meth:`transact`, which re-reads the journal under an exclusive lock.
    """

    def __init__(
        self,
        budgets: Mapping[str, object],
        *,
        journal_path: str | os.PathLike | None = None,
    ) -> None:
        self._journal_path = None if journal_path is None else os.fspath(journal_path)
        self._lock = threading.RLock()
        parsed: dict[str, Fraction] = {}
        for name, amount in budgets.items():
            if not isinstance(name, str) or not name:
                raise BudgetError("budget subject names must be non-empty strings")
            value = _nonneg_rational(amount, field=f"budget for {name!r}")
            parsed[name] = value
        self._initial = dict(parsed)
        self._remaining = dict(parsed)
        self._publications: dict[str, PublicationRecord] = {}
        if self._journal_path is not None:
            self._open_journal()

    # -- journal ---------------------------------------------------------------
    def _open_journal(self) -> None:
        path = self._journal_path
        assert path is not None
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Serialise header creation / first read against concurrent openers so a
        # second instance never observes a half-written journal.
        lock = self._file_lock()
        lock.__enter__()
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                header = {
                    "type": "header",
                    "format": LEDGER_FORMAT,
                    "initial_budgets": {
                        name: fmt_rational(value) for name, value in self._initial.items()
                    },
                }
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(_canonical(header) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                self._reload()
        finally:
            lock.__exit__(None, None, None)

    def _lock_path(self) -> str | None:
        return None if self._journal_path is None else self._journal_path + ".lock"

    def _reload(self) -> None:
        path = self._journal_path
        if path is None:
            return
        remaining = dict(self._initial)
        publications: dict[str, PublicationRecord] = {}
        header_seen = False
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("type") == "header":
                    header_seen = True
                    initial = {
                        name: Fraction(value)
                        for name, value in entry["initial_budgets"].items()
                    }
                    if initial != self._initial:
                        raise BudgetError(
                            "journal initial budgets do not match the configured budgets"
                        )
                    remaining = dict(initial)
                    continue
                if entry.get("type") != "publication":
                    raise BudgetError(f"malformed journal entry: {entry!r}")
                record = _record_from_journal(entry)
                publications[record.publication_id] = record
                for name, charge in record.charges.items():
                    remaining[name] = remaining.get(name, Fraction(0)) - charge
        if not header_seen:
            raise BudgetError("journal is missing its header")
        self._remaining = remaining
        self._publications = publications

    def _append(self, record: PublicationRecord) -> None:
        path = self._journal_path
        if path is None:
            return
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_canonical(record.to_journal()) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _file_lock(self):
        lock_path = self._lock_path()
        if lock_path is None or fcntl is None:
            return _NullContext()
        handle = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _LockedFile(handle)

    # -- introspection ---------------------------------------------------------
    def remaining(self, name: str) -> Fraction:
        with self._with_lock(reload=True):
            if name not in self._remaining:
                raise KeyError(name)
            return self._remaining[name]

    def remaining_all(self) -> dict[str, Fraction]:
        with self._with_lock(reload=True):
            return dict(self._remaining)

    def publication(self, publication_id: str) -> PublicationRecord | None:
        with self._with_lock(reload=True):
            return self._publications.get(publication_id)

    def audit(self) -> dict:
        """Recompute the ledger from the journal and report conservation."""

        with self._with_lock(reload=True):
            recomputed = dict(self._initial)
            for record in self._publications.values():
                for name, charge in record.charges.items():
                    recomputed[name] = recomputed.get(name, Fraction(0)) - charge
            conserved = recomputed == self._remaining
            nonnegative = all(value >= 0 for value in self._remaining.values())
            return {
                "conserved": conserved,
                "nonnegative": nonnegative,
                "ok": conserved and nonnegative,
                "initial": {name: fmt_rational(v) for name, v in self._initial.items()},
                "remaining": {
                    name: fmt_rational(v) for name, v in self._remaining.items()
                },
                "publications": len(self._publications),
            }

    # -- transaction -----------------------------------------------------------
    def _with_lock(self, *, reload: bool):
        ledger = self

        class _Ctx:
            def __enter__(self_inner):
                ledger._lock.acquire()
                try:
                    self_inner._file = ledger._file_lock()
                    self_inner._file.__enter__()
                    if reload:
                        ledger._reload()
                except BaseException:
                    if getattr(self_inner, "_file", None) is not None:
                        self_inner._file.__exit__(None, None, None)
                    ledger._lock.release()
                    raise
                return ledger

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    self_inner._file.__exit__(exc_type, exc, tb)
                finally:
                    ledger._lock.release()
                return False

        return _Ctx()

    def transact(self, publication_id: str, fingerprint: str) -> "_Transaction":
        """Begin a transaction; use as a context manager."""

        return _Transaction(self, publication_id, fingerprint)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _LockedFile:
    def __init__(self, handle) -> None:
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
        return False


class _Transaction:
    """An open ledger transaction.

    The transaction holds both the in-process lock and the file lock.  It
    re-reads the journal on entry so that a concurrent process's committed
    charge is visible before any budget decision is made.
    """

    def __init__(self, ledger: BudgetLedger, publication_id: str, fingerprint: str) -> None:
        self.ledger = ledger
        self.publication_id = publication_id
        self.fingerprint = fingerprint
        self._file = None
        self.existing: PublicationRecord | None = None

    def __enter__(self) -> "_Transaction":
        self.ledger._lock.acquire()
        try:
            self._file = self.ledger._file_lock()
            self._file.__enter__()
            self.ledger._reload()
        except BaseException:
            self._release()
            raise
        existing = self.ledger._publications.get(self.publication_id)
        if existing is not None and existing.fingerprint != self.fingerprint:
            # Release before raising so the lock is not held by the caller.
            self._release()
            raise PublicationConflict(
                f"publication id {self.publication_id!r} was already accepted with "
                "different input, source version, or scale"
            )
        self.existing = existing
        return self

    def __exit__(self, exc_type, exc, tb):
        self._release()
        return False

    def _release(self) -> None:
        try:
            if self._file is not None:
                self._file.__exit__(None, None, None)
                self._file = None
        finally:
            self.ledger._lock.release()

    # -- budget helpers --------------------------------------------------------
    def remaining(self, name: str) -> Fraction:
        return self.ledger._remaining.get(name, Fraction(0))

    def shortfalls(self, charges: Mapping[str, Fraction]) -> dict[str, dict]:
        short: dict[str, dict] = {}
        for name, charge in charges.items():
            have = self.remaining(name)
            if have < charge:
                short[name] = {
                    "charge": fmt_rational(charge),
                    "remaining": fmt_rational(have),
                    "shortfall": fmt_rational(charge - have),
                }
        return short

    def commit(
        self,
        *,
        charges: Mapping[str, Fraction],
        noised: Sequence[object],
        scale: Fraction,
        spec_digest: str,
        brief: dict,
    ) -> PublicationRecord:
        for name, charge in charges.items():
            if name not in self.ledger._remaining:
                self.ledger._remaining[name] = Fraction(0)
            self.ledger._remaining[name] -= charge
        record = PublicationRecord(
            publication_id=self.publication_id,
            fingerprint=self.fingerprint,
            charges=dict(charges),
            noised=tuple(noised),
            scale=scale,
            spec_digest=spec_digest,
            brief=brief,
        )
        self.ledger._publications[self.publication_id] = record
        self.ledger._append(record)
        return record


def _record_from_journal(entry: Mapping) -> PublicationRecord:
    try:
        return PublicationRecord(
            publication_id=str(entry["publication_id"]),
            fingerprint=str(entry["fingerprint"]),
            charges={name: Fraction(value) for name, value in entry["charges"].items()},
            noised=tuple(_decode_number(item) for item in entry["noised"]),
            scale=Fraction(entry["scale"]),
            spec_digest=str(entry["spec_digest"]),
            brief=entry["brief"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BudgetError(f"malformed publication record: {entry!r}") from exc


def _decode_number(item: Mapping) -> object:
    if "exact" in item:
        return Fraction(item["exact"])
    return float(item["float"])


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def laplace_noise_source(rng: random.Random | None = None) -> NoiseSource:
    """Return a source of independent zero-centred Laplace noise (float)."""

    generator = rng if rng is not None else random.Random()

    def _draw(count: int, scale: Fraction) -> list[float]:
        b = float(scale)
        out: list[float] = []
        for _ in range(count):
            # Inverse-CDF sample of Laplace(0, b) from a uniform on (-1/2, 1/2).
            u = generator.random() - 0.5
            magnitude = -b * math.log(1.0 - 2.0 * abs(u))
            out.append(math.copysign(magnitude, u) if u != 0.0 else 0.0)
        return out

    return _draw


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


def _nonneg_rational(value: object, *, field: str) -> Fraction:
    try:
        parsed = parse_rational(value, field=field)
    except RationalError as exc:
        raise BudgetError(str(exc)) from exc
    if parsed < 0:
        raise BudgetError(f"{field} must be non-negative")
    return parsed


def _positive_rational(value: object, *, field: str) -> Fraction:
    parsed = _nonneg_rational(value, field=field)
    if parsed <= 0:
        raise BudgetError(f"{field} must be strictly positive")
    return parsed


def _coefficient(noise: object) -> object:
    """Parse an injected noise value.

    ``Fraction``/``int``/decimal strings stay exact so tests can audit the
    drawing and the accounting separately; a ``float`` stays a float so a real
    sampler keeps its native arithmetic.
    """

    if isinstance(noise, bool):
        raise PublisherError("noise values must be numeric, not booleans")
    if isinstance(noise, Fraction):
        return noise
    if isinstance(noise, int):
        return Fraction(noise)
    if isinstance(noise, float):
        if math.isnan(noise) or math.isinf(noise):
            raise PublisherError("noise values must be finite")
        return noise
    try:
        return parse_rational(noise, field="noise")
    except RationalError as exc:
        raise PublisherError(str(exc)) from exc


class BriefPublisher:
    """Publish noisy histograms with persistent per-subject budgets.

    ``budgets`` maps subject name to its remaining allowance.  ``noise_source``
    is injectable for tests; by default each publication draws fresh,
    independent system randomness.
    """

    def __init__(
        self,
        budgets: Mapping[str, object],
        *,
        journal_path: str | os.PathLike | None = None,
        noise_source: NoiseSource | None = None,
    ) -> None:
        self.ledger = BudgetLedger(budgets, journal_path=journal_path)
        self._noise_source = noise_source or laplace_noise_source()

    # -- public API ------------------------------------------------------------
    def publish(
        self,
        publication_id: str,
        product: ContributionProduct,
        *,
        scale: object,
    ) -> dict:
        """Publish ``product`` under ``publication_id`` at rational ``scale``.

        Returns the external brief.  Raises :class:`PublicationRejected` when a
        budget is insufficient (nothing is charged, nothing is revealed),
        :class:`PublicationConflict` when the id was used with other inputs, and
        :class:`PublisherError` for malformed arguments.
        """

        if not isinstance(publication_id, str) or not publication_id.strip():
            raise PublisherError("publication_id must be a non-empty string")
        if not isinstance(product, ContributionProduct):
            raise PublisherError("product must be a ContributionProduct")
        b = _positive_rational(scale, field="scale")

        fingerprint = _sha256(
            {
                "product_digest": product.digest,
                "source": product.source.to_semantic(),
                "scale": fmt_rational(b),
            }
        )

        with self.ledger.transact(publication_id, fingerprint) as txn:
            if txn.existing is not None:
                # Idempotent re-read: same values, no extra charge.
                return _copy_brief(txn.existing.brief)

            charges = {
                state.name: state.replacement_bound / b
                for state in product.subject_states
            }
            short = txn.shortfalls(charges)
            if short:
                # Refuse the whole brief: nothing charged, no totals revealed.
                raise PublicationRejected(publication_id, short)

            noise_values = list(self._noise_source(product.spec.n_columns, b))
            if len(noise_values) != product.spec.n_columns:
                raise PublisherError(
                    "noise source returned "
                    f"{len(noise_values)} values for {product.spec.n_columns} columns"
                )
            noised: list[object] = []
            for total, noise in zip(product.column_sums, noise_values):
                noised.append(total + _coefficient(noise))

            brief = _external_brief(
                publication_id=publication_id,
                product=product,
                scale=b,
                noised=noised,
            )
            txn.commit(
                charges=charges,
                noised=noised,
                scale=b,
                spec_digest=product.spec.digest(),
                brief=brief,
            )
            return _copy_brief(brief)

    def brief(self, publication_id: str) -> dict:
        """Return the accepted external brief for ``publication_id``.

        Re-reading an accepted publication never charges again.
        """

        record = self.ledger.publication(publication_id)
        if record is None:
            raise KeyError(publication_id)
        return _copy_brief(record.brief)

    def internal_record(self, publication_id: str) -> dict:
        """Internal accounting view (charges, exact noise, fingerprint)."""

        record = self.ledger.publication(publication_id)
        if record is None:
            raise KeyError(publication_id)
        return {
            "publication_id": record.publication_id,
            "fingerprint": record.fingerprint,
            "charges": {name: fmt_rational(v) for name, v in record.charges.items()},
            "noised": [_encode_number(v) for v in record.noised],
            "scale": fmt_rational(record.scale),
            "spec_digest": record.spec_digest,
        }


def _external_brief(
    *,
    publication_id: str,
    product: ContributionProduct,
    scale: Fraction,
    noised: Sequence[object],
) -> dict:
    # Only public parameters and noisy totals leave the publisher.  The true
    # totals, current means and bound witnesses stay internal.
    return {
        "format": "brief/v1",
        "publication_id": publication_id,
        "spec_digest": product.spec.digest(),
        "source": product.source.to_semantic(),
        "scale": {"exact": fmt_rational(scale), "float": float(scale)},
        "replacement_bounds": {
            state.name: {
                "exact": fmt_rational(state.replacement_bound),
                "float": float(state.replacement_bound),
            }
            for state in product.subject_states
        },
        "columns": [
            {"name": column, "noised": _encode_number(value)}
            for column, value in zip(product.spec.columns, noised)
        ],
    }


def _copy_brief(brief: dict) -> dict:
    return json.loads(json.dumps(brief))

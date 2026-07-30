"""The memory pipeline: write stage, consolidation engine, mechanical decay.

This package is where the standard becomes a running system. Every stage
follows one discipline: anything an operator (LLM or heuristic) wants done
arrives as a typed proposal, is validated against the lifecycle machine, and
is rejected — never repaired — when invalid. Every mutation the pipeline
performs is recorded in a report; the audit trail is a first-class output.
"""

from __future__ import annotations

from .consolidate import (
    ConsolidationConfig,
    ConsolidationOperators,
    ConsolidationReport,
    HeuristicContradictionOperator,
    HeuristicResolutionOperator,
    TrigramNearDupOperator,
    consolidate,
    default_operators,
    resolve,
)
from .decay import DecayConfig, DecayReport, SalienceScores, decay, salience
from .report import Action, Rejection
from .write import IngestReport, ingest

__all__ = [
    "Action",
    "ConsolidationConfig",
    "ConsolidationOperators",
    "ConsolidationReport",
    "DecayConfig",
    "DecayReport",
    "HeuristicContradictionOperator",
    "HeuristicResolutionOperator",
    "IngestReport",
    "Rejection",
    "SalienceScores",
    "TrigramNearDupOperator",
    "consolidate",
    "decay",
    "default_operators",
    "ingest",
    "resolve",
    "salience",
]

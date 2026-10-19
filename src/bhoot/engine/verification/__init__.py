"""Verification: locate quotes on the page, check values inside them, compare answers."""

from bhoot.engine.verification.locator import LocatedSpan, QuoteLocator, has_text
from bhoot.engine.verification.matching import ValueComparator, ValueInQuoteMatcher
from bhoot.engine.verification.verifier import EvidenceVerifier

__all__ = [
    "EvidenceVerifier",
    "LocatedSpan",
    "QuoteLocator",
    "ValueComparator",
    "ValueInQuoteMatcher",
    "has_text",
]

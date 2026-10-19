"""bhoot: verified, grounded structured extraction from documents.

Every value comes with the page and quote it was read from and the checks it passed; anything
that can't be proven goes to review instead of being guessed.
"""

from bhoot._version import __version__
from bhoot.api import Bhoot, aextract, extract
from bhoot.domain.checks import CheckResult, CheckStatus, Severity
from bhoot.domain.locales import Locale
from bhoot.domain.modes import Mode
from bhoot.domain.options import ExtractOptions
from bhoot.domain.pack import ChunkSpec, DocPack, PageSelectorSpec, SectionSpec
from bhoot.domain.result import ExtractionResult, FieldResult, JobStatus
from bhoot.domain.rules import RuleContext, RuleScope, fail, pass_, rule, warn
from bhoot.domain.schema import Criticality, Granularity
from bhoot.domain.schema_dsl import Schema, field

__all__ = [
    "Bhoot",
    "CheckResult",
    "CheckStatus",
    "ChunkSpec",
    "Criticality",
    "DocPack",
    "ExtractOptions",
    "ExtractionResult",
    "FieldResult",
    "Granularity",
    "JobStatus",
    "Locale",
    "Mode",
    "PageSelectorSpec",
    "RuleContext",
    "RuleScope",
    "Schema",
    "SectionSpec",
    "Severity",
    "__version__",
    "aextract",
    "extract",
    "fail",
    "field",
    "pass_",
    "rule",
    "warn",
]

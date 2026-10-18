"""Exception hierarchy.

Developer mistakes (config, registry) are raised; document and provider problems are turned
into results with a status by the runner. Every error carries a plain message that says what
to do next where that is knowable.
"""

from __future__ import annotations


class BhootError(Exception):
    """Base class for every error raised by bhoot."""


# ---------------------------------------------------------------- configuration
class ConfigError(BhootError):
    """Invalid settings or configuration file."""


class ConfigFileError(ConfigError):
    """bhoot.yaml is invalid, has an unknown key, or contains schema definitions."""


class ModelConfigError(ConfigError):
    """A model tier is missing or misconfigured."""


class GraphSpecError(ConfigError):
    """Unknown node or router, or a dangling edge in a graph specification."""


class UnknownStrategyError(ConfigError):
    """A selector, chunker, merge strategy, coercer or policy name is not registered."""


class DuplicateRegistrationError(ConfigError):
    """Two strategies or rules were registered under the same name."""


# ---------------------------------------------------------------- registry
class RegistryError(BhootError):
    """Problems resolving schemas, prompts, packs or rules."""


class InvalidRefError(RegistryError):
    """A reference such as 'bank_statement@2' has invalid syntax."""


class RefNotFoundError(RegistryError):
    """No schema, prompt or pack with that name or version."""


class LabelNotFoundError(RegistryError):
    """The requested label does not exist."""


class ImmutableVersionModifiedError(RegistryError):
    """A migrated version's files were edited after the migration was written."""


class PackNotFoundError(RegistryError):
    """No installed pack with that name."""


class PackValidationError(RegistryError):
    """A pack is internally inconsistent (paths, strategies, rules, examples)."""


class RuleNotFoundError(RegistryError):
    """A rule id referenced by a schema or pack is not registered."""


class DuplicateDefinitionError(RegistryError):
    """Two different definitions were given the same name."""


class SchemaDriftError(RegistryError):
    """Strict mode: code definitions differ from the head migration."""


class BreakingChangeError(RegistryError):
    """A migration contains breaking changes and was not explicitly accepted."""


class MigrationDivergedError(RegistryError):
    """Two migrations share the same parent; history must be resolved."""


# ---------------------------------------------------------------- documents
class DocumentError(BhootError):
    """The input document cannot be processed."""


class UnsupportedFormatError(DocumentError):
    """The file type is not supported."""


class EncryptedPdfError(DocumentError):
    """The PDF is encrypted and no (or a wrong) password was given."""


class CorruptPdfError(DocumentError):
    """The PDF cannot be parsed."""


class FileTooLargeError(DocumentError):
    """The file exceeds the configured size limit."""


class TooManyPagesError(DocumentError):
    """The document exceeds the configured page limit."""


# ---------------------------------------------------------------- jobs
class JobError(BhootError):
    """Problems with a job as a whole."""


class JobNotFoundError(JobError):
    """No checkpoint exists for the job id (resume)."""


class IncompatibleCheckpointError(JobError):
    """The checkpoint was written by an incompatible engine version."""


class JobTimeoutError(JobError):
    """The job exceeded its time limit."""


# ---------------------------------------------------------------- LLM providers
class LLMError(BhootError):
    """Problems calling a model provider."""


class ProviderAuthError(LLMError):
    """Authentication with the provider failed. Not retried."""


class ProviderTransientError(LLMError):
    """Timeouts, connection errors, 5xx responses. Retried with backoff."""


class RateLimitedError(LLMError):
    """The provider rate-limited the request. Retried with backoff."""


class ModelCapabilityError(LLMError):
    """The model lacks a needed capability (e.g. vision, JSON schema)."""


class ResponseParseError(LLMError):
    """The model's answer is not valid JSON for the requested schema."""


# ---------------------------------------------------------------- engine
class EngineError(BhootError):
    """An unexpected failure inside the engine, wrapped with the node name."""

    def __init__(self, message: str, *, node: str | None = None) -> None:
        super().__init__(message)
        self.node = node


class IllegalTransitionError(EngineError):
    """A field status change that the state machine does not allow (a bug guard)."""


class EvalError(BhootError):
    """Problems running an evaluation."""

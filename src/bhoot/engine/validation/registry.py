"""Rule registry: rules from packs and rules attached by id."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from bhoot.domain.errors import DuplicateRegistrationError, RuleNotFoundError
from bhoot.domain.pack import DocPack
from bhoot.domain.rules import Rule


class RuleRegistry:
    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self._rules: dict[str, Rule] = {}
        for r in rules:
            self.register(r)

    def register(self, r: Rule) -> None:
        existing = self._rules.get(r.rule_id)
        if existing is not None and existing is not r:
            raise DuplicateRegistrationError(f"rule {r.rule_id!r} is already registered")
        self._rules[r.rule_id] = r

    def get(self, rule_id: str) -> Rule:
        try:
            return self._rules[rule_id]
        except KeyError:
            raise RuleNotFoundError(
                f"no rule {rule_id!r}; registered: {sorted(self._rules)}"
            ) from None

    def rules_for(self, pack: DocPack, attached: Sequence[str] = ()) -> tuple[Rule, ...]:
        """The pack's own rules plus rules attached by id, each once, pack rules first."""
        out: dict[str, Rule] = {r.rule_id: r for r in pack.rules}
        for rule_id in attached:
            out.setdefault(rule_id, self.get(rule_id))
        return tuple(out.values())

"""A unit glyph must never be uppercased by CSS.

Unicode maps µ (U+00B5 MICRO SIGN) to Μ (U+039C GREEK CAPITAL MU), which
renders glyph-identical to a Latin M. So `text-transform: uppercase` on an
ancestor turns "1970 µm to floor" into "1970 ΜM TO FLOOR" — a reader sees
millimetres where the code means microns. That shipped on the
sample-at-objective banner, invisible in review because the markup is correct:
the bug lives in the CSS of an ancestor, in a different file.

`text-transform` is inherited, so the value that reaches a text node comes from
the nearest element in its ancestor chain that declares one. This walks the
real templates against the real stylesheets and resolves exactly that, so a new
µ under a new uppercased panel fails here instead of on the microscope.

ponytail: the matcher handles descendant combinators only (`.a b`, `.a .b`).
Pseudo-elements are ignored on purpose — generated content is never a template
text node. Anything else it cannot model (child/sibling combinators,
pseudo-classes, id and attribute selectors) is reported by
test_matcher_covers_every_text_transform_rule rather than silently skipped, so
a hole in the guard fails the suite instead of hiding in it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
MICRO_SIGNS = ("µ", "μ")  # MICRO SIGN, GREEK SMALL LETTER MU

# Elements that never nest and so never carry a text child.
VOID_TAGS = frozenset({"br", "img", "input", "hr", "meta", "link", "source", "area", "base", "col"})

UNSUPPORTED_COMBINATOR = re.compile(r"[>+~]|::|:(?!root\b)")


class Compound(NamedTuple):
    """One simple selector: an optional tag name plus any classes."""

    tag: str | None
    classes: frozenset[str]

    def matches(self, tag: str, classes: frozenset[str]) -> bool:
        if self.tag is not None and self.tag != tag:
            return False
        return self.classes <= classes


class Rule(NamedTuple):
    compounds: tuple[Compound, ...]  # descendant chain, target last
    value: str
    source: str

    @property
    def weight(self) -> int:
        """Rough specificity: enough to order rules matching the same element."""
        return sum(len(c.classes) * 10 + (1 if c.tag else 0) for c in self.compounds)


class Element(NamedTuple):
    tag: str
    classes: frozenset[str]


def _parse_compound(text: str) -> Compound | None:
    tag_match = re.match(r"^([A-Za-z][A-Za-z0-9]*)", text)
    tag = tag_match.group(1) if tag_match else None
    rest = text[len(tag) :] if tag else text
    classes = re.findall(r"\.([A-Za-z0-9_-]+)", rest)
    leftover = re.sub(r"\.[A-Za-z0-9_-]+", "", rest).strip()
    if leftover or (tag is None and not classes):
        return None  # id, attribute, universal, or something we do not model
    return Compound(tag, frozenset(classes))


def text_transform_rules() -> tuple[list[Rule], list[str]]:
    """Every text-transform rule in the stylesheets, plus the ones we skipped."""
    rules: list[Rule] = []
    skipped: list[str] = []
    for sheet in sorted((WEB / "static" / "css").rglob("*.css")):
        # Comments first: prose is full of commas and braces, and this repo
        # comments heavily. Left in, it parses as selectors.
        css = re.sub(r"/\*.*?\*/", "", sheet.read_text(), flags=re.DOTALL)
        for block in re.finditer(r"([^{}]+)\{([^}]*)\}", css):
            selector_group, body = block.group(1), block.group(2)
            declared = re.search(r"text-transform\s*:\s*([A-Za-z-]+)", body)
            if not declared:
                continue
            value = declared.group(1).lower()
            for selector in selector_group.split(","):
                selector = selector.strip()
                if not selector:
                    continue
                if "::before" in selector or "::after" in selector:
                    continue  # generated content, never a template text node
                if UNSUPPORTED_COMBINATOR.search(selector):
                    skipped.append(f"{sheet.name}: {selector}")
                    continue
                compounds = [_parse_compound(part) for part in selector.split()]
                if not compounds or any(c is None for c in compounds):
                    skipped.append(f"{sheet.name}: {selector}")
                    continue
                rules.append(
                    Rule(tuple(c for c in compounds if c), value, f"{sheet.name}: {selector}")
                )
    return rules, skipped


def resolve_text_transform(chain: list[Element], rules: list[Rule]) -> tuple[str, str] | None:
    """The inherited text-transform reaching the innermost element of `chain`.

    Walks outward from the text node's parent; the nearest element carrying a
    declaration wins, which is how an inherited property actually resolves.
    """
    for depth in range(len(chain), 0, -1):
        target = chain[depth - 1]
        winner: Rule | None = None
        for rule in rules:
            if not rule.compounds[-1].matches(target.tag, target.classes):
                continue
            # Remaining compounds must match ancestors, in order.
            ancestors = list(chain[: depth - 1])
            needed = list(rule.compounds[:-1])
            while needed and ancestors:
                candidate = ancestors.pop()
                if needed[-1].matches(candidate.tag, candidate.classes):
                    needed.pop()
            if needed:
                continue
            if winner is None or rule.weight >= winner.weight:
                winner = rule
        if winner is not None:
            return winner.value, winner.source
    return None


class MicronFinder(HTMLParser):
    """Collect µ-bearing text nodes together with their full ancestor chain."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[Element] = []
        self.hits: list[tuple[int, list[Element], str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in VOID_TAGS:
            return
        classes = frozenset((dict(attrs).get("class") or "").split())
        self._stack.append(Element(tag, classes))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if any(sign in data for sign in MICRO_SIGNS):
            self.hits.append((self.getpos()[0], list(self._stack), data.strip()))


def test_no_micron_is_uppercased() -> None:
    rules, _ = text_transform_rules()
    assert rules, "found no text-transform rules at all — the scan is broken, not the templates"

    offences: list[str] = []
    for template in sorted((WEB / "templates").rglob("*.html")):
        finder = MicronFinder()
        finder.feed(template.read_text())
        for line, chain, text in finder.hits:
            resolved = resolve_text_transform(chain, rules)
            if resolved and resolved[0] == "uppercase":
                offences.append(
                    f"{template.relative_to(WEB)}:{line}: {text!r} is uppercased by "
                    f"[{resolved[1]}] — µ uppercases to Μ and reads as M. Set "
                    f"text-transform: none on the element holding the unit."
                )

    assert not offences, "\n".join(offences)


def test_matcher_covers_every_text_transform_rule() -> None:
    """Skipped selectors are a hole in the guard, so keep the list visible."""
    _, skipped = text_transform_rules()
    assert not skipped, (
        "these text-transform rules use selector syntax the matcher does not model, "
        "so a µ under them would go unchecked:\n" + "\n".join(sorted(set(skipped)))
    )


def test_the_guard_fails_on_the_shape_it_guards() -> None:
    """Worth having only if it catches the banner bug when the reset is removed."""
    chain = [
        Element("span", frozenset({"op-lock-txt"})),
        Element("i", frozenset()),
    ]
    uppercase_ancestor = Rule((Compound("span", frozenset({"op-lock-txt"})),), "uppercase", "test")
    reset_on_unit = Rule(
        (Compound(None, frozenset({"op-lock-txt"})), Compound("i", frozenset())), "none", "test"
    )

    without_reset = resolve_text_transform(chain, [uppercase_ancestor])
    assert without_reset is not None and without_reset[0] == "uppercase"

    with_reset = resolve_text_transform(chain, [uppercase_ancestor, reset_on_unit])
    assert with_reset is not None and with_reset[0] == "none"

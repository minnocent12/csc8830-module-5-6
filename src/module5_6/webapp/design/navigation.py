"""Navigation model for the app shell: stable slugs and URL resolution. No Streamlit here.

Pages are any objects with the shared structural contract ``module_label``, ``page_label``,
``order``, and ``render`` (each assignment defines its own ``PageSpec``). The model groups
them by module in the order given, which callers have already sorted.

Slugs are the canonical URL form, ``?module=module-5-6&page=motion-tracking``: lowercase
ASCII words joined by single hyphens, with ``&`` and every other non-alphanumeric run
treated as a separator. They are deterministic, so links stay valid across sessions and
deployments. Duplicate slugs are an error, never resolved by picking one silently.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Reserved for the dashboard Home view (a later phase); no module may take this slug.
RESERVED_MODULE_SLUGS = frozenset({"home"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(label: str) -> str:
    """``"Experiments & Results"`` becomes ``"experiments-results"``."""
    ascii_text = unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode()
    slug = _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")
    if not slug:
        raise ValueError(f"label {label!r} has no characters usable in a URL slug")
    return slug


@dataclass(frozen=True)
class NavPage:
    """One registered page with its stable slug."""

    label: str
    slug: str
    page: Any


@dataclass(frozen=True)
class NavModule:
    """One module and its pages, in registration order."""

    label: str
    slug: str
    pages: tuple[NavPage, ...]

    def page_by_slug(self, slug: str | None) -> NavPage | None:
        return next((p for p in self.pages if p.slug == slug), None)

    def page_by_label(self, label: str | None) -> NavPage | None:
        return next((p for p in self.pages if p.label == label), None)


def build_navigation(pages: Sequence[Any]) -> tuple[NavModule, ...]:
    """Group pages by module, preserving order, and assign slugs.

    Raises ``ValueError`` when two modules, or two pages within one module, share a slug,
    or when a module would take a reserved slug.
    """
    grouped: dict[str, list[Any]] = {}
    for page in pages:
        grouped.setdefault(str(page.module_label), []).append(page)

    modules: list[NavModule] = []
    module_slugs: dict[str, str] = {}
    for module_label, module_pages in grouped.items():
        slug = slugify(module_label)
        if slug in RESERVED_MODULE_SLUGS:
            raise ValueError(f"module {module_label!r} would use the reserved slug {slug!r}")
        if slug in module_slugs:
            raise ValueError(
                f"modules {module_slugs[slug]!r} and {module_label!r} share the slug {slug!r}"
            )
        module_slugs[slug] = module_label
        nav_pages: list[NavPage] = []
        page_slugs: dict[str, str] = {}
        for page in module_pages:
            label = str(page.page_label)
            page_slug = slugify(label)
            if page_slug in page_slugs:
                raise ValueError(
                    f"pages {page_slugs[page_slug]!r} and {label!r} in {module_label!r} "
                    f"share the slug {page_slug!r}"
                )
            page_slugs[page_slug] = label
            nav_pages.append(NavPage(label, page_slug, page))
        modules.append(NavModule(module_label, slug, tuple(nav_pages)))
    return tuple(modules)


def resolve(
    modules: Sequence[NavModule], module_slug: str | None, page_slug: str | None
) -> tuple[NavModule, NavPage]:
    """Resolve URL slugs to a module and page, never failing.

    * unknown or missing module: the first module and its first page
    * known module, unknown or missing page: that module's first page, never a page from
      another module
    """
    if not modules:
        raise ValueError("no pages are registered")
    module = next((m for m in modules if m.slug == module_slug), None)
    if module is None:
        return modules[0], modules[0].pages[0]
    return module, module.page_by_slug(page_slug) or module.pages[0]


def module_by_label(modules: Sequence[NavModule], label: str | None) -> NavModule | None:
    return next((m for m in modules if m.label == label), None)

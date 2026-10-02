"""Public copy checks: no em or en dash and no internal references in anything the app shows.

Three layers, each scoped to text that can reach users:
1. Web app source files contain no em or en dash at all (comments included).
2. Every string literal in the package (docstrings excluded) is dash-free; web app string
   literals also contain no repository paths, document names, or phase names, except the
   allowlisted file-loading constants below, which are never displayed.
3. Documents rendered into pages, and the text of every rendered page, pass the same rules.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "src" / "module5_6"
WEBAPP = PACKAGE / "webapp"
APP_PATH = str(REPO_ROOT / "app.py")
DASHES = (chr(0x2013), chr(0x2014))  # en dash, em dash
INTERNAL = re.compile(
    r"IMPLEMENTATION_PLAN|EXPERIMENTAL_RESULTS|\b[A-Za-z_]+\.md\b|\bdocs/|\bresults/|\bdata/|"
    r"\bscripts/|\bsrc/|\.py\b|foundation phase|later phase|\bPhase \d|professor-required",
    re.IGNORECASE,
)
# File-loading constants: they name files the code opens and are never displayed.
NOT_DISPLAYED = {"results/tracking/video_1/P1_record.json", "results/tracking/video_2/P1_record.json", "results/optical_flow/video_1/video_1_optical_flow_summary.json", "results/optical_flow/video_2/video_2_optical_flow_summary.json", "results/sfm/sfm_summary.json", "results/sfm/{}_orb_keypoints.jpg"}
# Documents shown verbatim on a page.
RENDERED_DOCS = []
# A user-facing artifact name the app itself produces and asks for, not a repository path.
ALLOWED_IN_RENDERED_TEXT = re.compile(r"calibration\.json")


def _webapp_files() -> list[Path]:
    return sorted(p for p in WEBAPP.glob("*.py"))


def _literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    parts = {id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            found.append((node.lineno, "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docstrings and id(node) not in parts):
            found.append((node.lineno, node.value))
    return found


@pytest.mark.parametrize("path", _webapp_files(), ids=lambda p: p.name)
def test_web_app_sources_have_no_em_or_en_dash(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not any(d in text for d in DASHES), path.name


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_string_literals_have_no_em_or_en_dash(path: Path) -> None:
    bad = [(line, text[:80]) for line, text in _literals(path) if any(d in text for d in DASHES)]
    assert not bad, bad


@pytest.mark.parametrize("path", _webapp_files(), ids=lambda p: p.name)
def test_web_app_literals_have_no_internal_references(path: Path) -> None:
    bad = [
        (line, text[:100]) for line, text in _literals(path)
        if INTERNAL.search(text) and text not in NOT_DISPLAYED
    ]
    assert not bad, bad


@pytest.mark.parametrize("relative", RENDERED_DOCS)
def test_rendered_documents_are_public_copy(relative: str) -> None:
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert not any(d in text for d in DASHES), relative
    assert not INTERNAL.findall(text), INTERNAL.findall(text)


def _visible_text(app: AppTest) -> list[str]:
    texts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "success", "error"):
        texts += [str(e.value) for e in getattr(app, kind)]
    texts += [m.label for m in app.metric] + [str(m.value) for m in app.metric]
    texts += [e.label for e in app.expander]
    for kind in ("button", "checkbox", "radio", "selectbox", "slider", "select_slider", "number_input", "text_input"):
        for w in getattr(app, kind):
            texts += [str(w.label), str(getattr(w.proto, "help", ""))]
    texts += [e.proto.body for e in app.get("html") if "<style" not in e.proto.body]
    return texts


def test_every_rendered_page_is_public_copy() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    problems = []
    for page in list(app.sidebar.radio[0].options):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        for text in _visible_text(app):
            if any(d in text for d in DASHES):
                problems.append((page, "dash", text[:80]))
            refs = [r for r in INTERNAL.findall(ALLOWED_IN_RENDERED_TEXT.sub("", text))]
            if refs:
                problems.append((page, refs, text[:80]))
    assert not problems, problems

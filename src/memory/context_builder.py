# src/memory/context_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Set

try:
    from llama_index.core.schema import NodeWithScore
except Exception:  # pragma: no cover
    NodeWithScore = None  # type: ignore


@dataclass
class ContextConfig:
    """
    Configuration for building LLM-ready context.
    - max_chars: hard budget for total characters in the context block
    - max_items_per_ns: limit of included items per namespace
    - include_metadata_keys: metadata keys to surface inline (order matters)
    - heading_sales/insta: section titles; override if you localize
    - dedupe_on: metadata keys used to deduplicate (e.g., date+product_id)
    """
    max_chars: int = 8000
    max_items_per_ns: int = 8
    include_metadata_keys: Tuple[str, ...] = ("date", "product_id", "post_id", "media_type", "period")
    heading_sales: str = "Sales Evidence"
    heading_insta: str = "Instagram Evidence"
    heading_other: str = "Other Evidence"
    dedupe_on: Tuple[str, ...] = ("namespace", "date", "product_id", "post_id")


def build(
    query: str,
    items: List,  # List[NodeWithScore] in runtime
    cfg: Optional[ContextConfig] = None,
) -> str:
    """
    Build a structured context string for the LLM from retrieved ScoredItems.
    The result is deterministic given inputs and budget limits.

    :param query: the user/system query (added as a header for clarity)
    :param items: retrieved and (optionally) fused/reranked ScoredItems
    :param cfg: ContextConfig for budgeting and formatting
    :return: context string
    """
    cfg = cfg or ContextConfig()

    # 1) Group by namespace (e.g., "sales", "insta", ...)
    groups: Dict[str, List] = _group_by_namespace(items)

    # 2) Dedupe + cap per namespace
    for ns, lst in groups.items():
        groups[ns] = _dedupe_items(lst, cfg.dedupe_on)[: cfg.max_items_per_ns]

    # 3) Render sections
    sections: List[str] = []
    sections.append(_render_header(query))
    for ns in sorted(groups.keys()):
        title = _pick_heading(ns, cfg)
        rendered = _render_namespace_section(ns, title, groups[ns], cfg)
        if rendered:
            sections.append(rendered)

    # 4) Apply global char budget
    context = _fit_budget("\n\n".join(sections), cfg.max_chars)
    return context


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_header(query: str) -> str:
    """Render a small header with the original query."""
    q = query.strip().replace("\n", " ").strip()
    return f"# Query\n{q}\n"


def _pick_heading(ns: str, cfg: ContextConfig) -> str:
    """Pick a human-friendly section title for a namespace."""
    if ns.lower() == "sales":
        return cfg.heading_sales
    if ns.lower() == "insta":
        return cfg.heading_insta
    return cfg.heading_other


def _render_namespace_section(
    namespace: str,
    title: str,
    items: List,  # List[NodeWithScore]
    cfg: ContextConfig,
) -> str:
    """
    Render a markdown section for a namespace.
    Each item is a bullet with concise metadata + content snippet + citation key.
    """
    if not items:
        return ""

    lines: List[str] = [f"## {title}"]
    for idx, si in enumerate(items, start=1):
        node = getattr(si, "node", None) or getattr(si, "item", None)
        text = getattr(node, "text", None) or getattr(node, "get_text", lambda: "")()
        content = (text or "").strip()
        meta_line = _format_metadata_inline(si, cfg.include_metadata_keys)
        cite = _format_citation(si, namespace, idx)
        # compact one-liner with short content preview
        preview = _shorten(content, 560)  # keep bullets tight
        lines.append(f"- {meta_line} {preview} {cite}")

    return "\n".join(lines)


def _format_metadata_inline(si, keys: Iterable[str]) -> str:
    """
    Format selected metadata keys inline in a stable order.
    Missing keys are skipped. Output example:
    [date=1404-05-12 | product_id=P123 | post_id=ABC]
    """
    node = getattr(si, "node", None) or getattr(si, "item", None)
    md = getattr(node, "metadata", {}) or {}
    chunks: List[str] = []
    for k in keys:
        if k in md and md[k] not in (None, ""):
            chunks.append(f"{k}={md[k]}")
    return f"[{ ' | '.join(chunks) }]" if chunks else "[]"


def _format_citation(si, namespace: str, idx: int) -> str:
    """
    Produce a compact, stable citation token that an LLM can echo back.
    Encodes namespace and a few key metadata fields.
    Example: (ref:sales|date=1404-05-12|product=P123|post=XYZ)
    """
    node = getattr(si, "node", None) or getattr(si, "item", None)
    md = getattr(node, "metadata", {}) or {}
    date = md.get("date", "")
    prod = md.get("product_id", "")
    post = md.get("post_id", "")
    # idx acts as a stable anchor within the section
    return f"(ref:{namespace}|#{idx}|date={date}|product={prod}|post={post})"


def _shorten(text: str, max_len: int) -> str:
    """Return a shortened (single-line) snippet suitable for bullets."""
    s = " ".join(text.split())
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")


def _fit_budget(text: str, max_chars: int) -> str:
    """
    Enforce a hard character budget on the final context.
    Truncates cleanly at the nearest paragraph boundary if possible.
    """
    if len(text) <= max_chars:
        return text
    # try to cut at a paragraph boundary within budget
    cut = text[:max_chars]
    last_break = cut.rfind("\n\n")
    if last_break > 0:
        return cut[:last_break].rstrip()
    return cut.rstrip()


# ---------------------------------------------------------------------------
# Grouping / Deduplication
# ---------------------------------------------------------------------------

def _group_by_namespace(items: List) -> Dict[str, List]:
    """Group results by item.namespace."""
    out: Dict[str, List] = {}
    for si in items:
        node = getattr(si, "node", None) or getattr(si, "item", None)
        md = getattr(node, "metadata", {}) or {}
        ns = str(md.get("namespace", "")).strip().lower()
        out.setdefault(ns, []).append(si)
    return out


def _dedupe_items(items: List, dedupe_on: Tuple[str, ...]) -> List:
    """
    Deduplicate items using a compound key built from:
      - 'namespace' special case
      - metadata fields listed in dedupe_on
      - a prefix of content to avoid repeated text
    The first occurrence is kept (assuming items are pre-sorted by relevance).
    """
    seen: Set[str] = set()
    output: List = []

    for si in items:
        node = getattr(si, "node", None) or getattr(si, "item", None)
        md = getattr(node, "metadata", {}) or {}
        parts: List[str] = []
        for k in dedupe_on:
            if k == "namespace":
                ns = md.get("namespace")
                if ns is None:
                    # Try legacy attribute on item
                    ns = getattr(getattr(si, "item", None), "namespace", "")
                parts.append(str(ns).lower())
            else:
                parts.append(str(md.get(k, "")))
        text = getattr(node, "text", None) or getattr(node, "get_text", lambda: "")()
        parts.append((text or "")[:80])
        key = "|".join(parts)

        if key in seen:
            continue
        seen.add(key)
        output.append(si)

    return output


# ---------------------------------------------------------------------------
# Optional: utility to build a final prompt block
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTIONS = (
    "You are a meticulous analyst. Use the evidence below to answer. "
    "Cite references by echoing their (ref:...) tokens when making claims. "
    "If information is insufficient, say so explicitly."
)

def build_prompt(
    query: str,
    items: List[ScoredItem],
    cfg: Optional[ContextConfig] = None,
    system_instructions: str = DEFAULT_INSTRUCTIONS,
) -> str:
    """
    Build a full prompt block: system-style instructions + structured context + question.
    This is a convenience wrapper around `build(...)`.
    """
    context = build(query=query, items=items, cfg=cfg)
    return (
        f"[INSTRUCTIONS]\n{system_instructions}\n\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[TASK]\nAnswer the query using the context above. Provide clear reasoning and reference tokens."
    )

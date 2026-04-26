"""扫描 vault，解析每张卡片的 frontmatter / 摘要 / wikilinks。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")
META_LINE_RE = re.compile(r"^\*\*[^*\n]+\*\*\s*[:：]")  # 匹配所有 **xxx**: / **xxx**：

VAULT_SUBDIRS = ["0-inbox", "1-concepts", "2-explorations", "3-projects", "4-references"]

INDEX_PRODUCT_NAMES = {"INDEX", "CLUSTERS", "GRAPH", "TERRAIN"}


@dataclass
class Card:
    path: Path
    name: str
    subdir: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    links_out: list[str] = field(default_factory=list)
    raw_links: list[str] = field(default_factory=list)


def normalize_link(target: str) -> str:
    """把 [[A]] / [[A.md]] / [[A|alias]] / [[A#section]] 统一成卡片名。"""
    target = target.strip()
    if "|" in target:
        target = target.split("|", 1)[0].strip()
    if "#" in target:
        target = target.split("#", 1)[0].strip()
    if target.endswith(".md"):
        target = target[:-3]
    return target


def extract_summary(content: str, fm_summary) -> str:
    """优先级：frontmatter.summary > '## 定义' 后第一句 > 第一段非空文本。"""
    if fm_summary:
        return str(fm_summary).strip()[:120]

    # 找 ## 定义 后第一句
    def_match = re.search(r"##\s*定义\s*\n+([^\n]+)", content)
    if def_match:
        first = def_match.group(1).strip()
        # 取到第一个句号
        sentence_match = re.match(r"^([^。.！？!?\n]+[。.！？!?]?)", first)
        if sentence_match:
            return sentence_match.group(1).strip()[:120]
        return first[:120]

    # 找第一段非空文本（跳过 frontmatter / 标题 / 图片 / 引用）
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "```", ">", "![", "---", "|")):
            continue
        if META_LINE_RE.match(line):
            continue
        return line[:120]
    return ""


def parse_card(path: Path, vault_root: Path) -> Card | None:
    try:
        post = frontmatter.load(path)
    except Exception:
        return None

    rel = path.relative_to(vault_root)
    subdir = rel.parts[0] if len(rel.parts) > 1 else ""
    name = path.stem

    if name in INDEX_PRODUCT_NAMES:
        return None  # 跳过自己生成的产物

    tags = post.metadata.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t).strip() for t in tags if str(t).strip()]

    raw_matches = WIKILINK_RE.findall(post.content)
    links_out = []
    seen = set()
    for raw in raw_matches:
        normalized = normalize_link(raw)
        if normalized and normalized != name and normalized not in seen:
            seen.add(normalized)
            links_out.append(normalized)

    summary = extract_summary(post.content, post.metadata.get("summary"))

    return Card(
        path=path,
        name=name,
        subdir=subdir,
        tags=tags,
        summary=summary,
        links_out=links_out,
        raw_links=raw_matches,
    )


def scan_vault(vault_root: Path) -> list[Card]:
    cards: list[Card] = []
    for sub in VAULT_SUBDIRS:
        d = vault_root / sub
        if not d.is_dir():
            continue
        for md in sorted(d.rglob("*.md")):
            card = parse_card(md, vault_root)
            if card:
                cards.append(card)
    return cards

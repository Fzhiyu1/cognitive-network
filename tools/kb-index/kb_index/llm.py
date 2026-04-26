"""LLM 摘要生成（Claude Haiku 4.5），写回卡片 frontmatter。"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from .parse import Card

MODEL = "claude-haiku-4-5"

SUMMARY_SYSTEM = """你是一个知识库摘要助手。给定一张概念/探索/项目卡片，用 1 句话（不超过 60 字）总结其核心观点。

要求：
- 中文输出
- 不要复述卡片标题
- 不要前言/后缀（如"这张卡讲的是…"）
- 不用引号包裹
- 直接输出摘要这一行内容，不输出别的"""

SUMMARY_USER = """卡片标题：{name}

卡片内容（截断到前 2000 字）：
{content}

输出："""


@dataclass
class LLMConfig:
    api_key: str
    model: str = MODEL
    max_chars_in: int = 2000
    max_tokens_out: int = 120


def get_client(cfg: LLMConfig | None = None):
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: 需要安装 anthropic SDK：pip install 'kb-index[llm]'", file=sys.stderr)
        sys.exit(2)
    cfg = cfg or LLMConfig(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    if not cfg.api_key:
        print("ERROR: 未找到 ANTHROPIC_API_KEY 环境变量", file=sys.stderr)
        print("       export ANTHROPIC_API_KEY=sk-ant-xxx 后重试", file=sys.stderr)
        sys.exit(2)
    return Anthropic(api_key=cfg.api_key), cfg


def summarize_card(client, cfg: LLMConfig, card: Card) -> str:
    """调 Claude 给一张卡生成摘要。返回干净的一句话。"""
    content = card.path.read_text(encoding="utf-8")
    # 去掉 frontmatter
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end > 0:
            content = content[end + 4:].strip()
    content = content[:cfg.max_chars_in]

    msg = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens_out,
        system=SUMMARY_SYSTEM,
        messages=[{
            "role": "user",
            "content": SUMMARY_USER.format(name=card.name, content=content),
        }],
    )

    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    text = text.strip('"\'""''「」『』')
    text = re.sub(r"\s+", " ", text)
    return text[:120]


def write_summary_to_frontmatter(card: Card, summary: str) -> None:
    """把 summary 写进卡片 frontmatter。保留原有所有字段。"""
    post = frontmatter.load(card.path)
    post.metadata["summary"] = summary
    # 用 frontmatter 库的写法保持兼容性
    serialized = frontmatter.dumps(post, sort_keys=False)
    # frontmatter 库默认输出尾部不带换行，对齐文件原本结构
    if not serialized.endswith("\n"):
        serialized += "\n"
    card.path.write_text(serialized, encoding="utf-8")


def fill_missing_summaries(
    cards: list[Card],
    *,
    write_to_card: bool = True,
    only_missing: bool = True,
    progress: bool = True,
) -> dict[str, str]:
    """对所有缺 summary 的卡（或全部）调 LLM。返回 {name: summary}。"""
    client, cfg = get_client()
    results: dict[str, str] = {}

    targets = []
    for card in cards:
        post = frontmatter.load(card.path)
        existing = post.metadata.get("summary")
        if only_missing and existing:
            continue
        targets.append(card)

    if progress:
        print(f"待生成: {len(targets)} 张卡（共 {len(cards)} 张）")

    for i, card in enumerate(targets, 1):
        try:
            summary = summarize_card(client, cfg, card)
            results[card.name] = summary
            if write_to_card:
                write_summary_to_frontmatter(card, summary)
            if progress:
                print(f"  [{i:3}/{len(targets)}] {card.name[:40]:40}  →  {summary[:60]}")
        except Exception as e:
            if progress:
                print(f"  [{i:3}/{len(targets)}] {card.name[:40]:40}  ✗ {e}")

    return results

# Cognitive Network

输出驱动的个人认知网络 — 记录的不是信息，是思考过程和思考产物。

## 这是什么

一个基于 Zettelkasten 卡片盒笔记法的个人知识库，用 Obsidian 管理，通过双向链接构建概念之间的网络。

## 结构

```
0-inbox/        # 未整理的想法
1-concepts/     # 原子化概念卡片（一张一个概念）
2-explorations/ # 完整的探索记录（对话整理）
3-projects/     # 有明确行动计划的项目
4-references/   # 参考资料
```

## 收录标准

核心问题：**这个内容里有没有"我的思考"？**

有我的思考才收录。复制粘贴的教程、未经加工的收藏、纯摘抄不收录。

详见 `1-concepts/知识库元认知.md`。

## 使用方式

1. 用 [Obsidian](https://obsidian.md/) 打开本仓库
2. 概念卡片之间通过 `[[双向链接]]` 互相连接
3. 用图谱视图查看概念网络

## 附带工具

`.claude/skills/knowledge-base-watcher/` — 一个 Claude Code skill，在对话中检测到新概念时主动询问是否入库。应安装在用户级别（`~/.claude/skills/`）。

"""M9：最小 RAG —— 把"外部知识"变成"可以塞进上下文的一段证据"。

RAG 五个环节（缺一个都会出问题）::

    文档 ──①切块──▶ chunks ──②向量化──▶ vectors
                                            │
    用户问题 ──②向量化──▶ query 向量 ──③相似度排序──▶ top-k
                                            │
                                     ④阈值过滤（宁可说"没找到"）
                                            │
                                     ⑤带引用的证据 → 塞进 prompt

三条设计原则：

    A. **确定性**：用"哈希向量化 + IDF 加权"，不依赖真实 embedding 模型。
       同一段文字永远得到同一个向量——测试和回放才可靠。
       （真实项目换成 embedding API，接口不用改。）
    B. **拒答优先**：相似度低于阈值就返回 found=False。
       宁可说"没找到相关条款"，也不能把最不相关的片段当答案。
    C. **可追溯**：每条结果都带 doc_id + chunk_index + 原文。

关于哈希向量化的两个已知坑（本模块已处理）：

    1. **哈希碰撞抵消**：不同 token 落到同一维度、符号相反时会互相抵消，
       让一个真实存在的关键词得分变成 0。→ 用更大维度 + IDF 加权缓解。
    2. **垃圾 token**：跨标点拼接会产生 "送与"、"与运" 这类噪声。
       → 按"片段"（连续的汉字串 / 连续的字母数字串）分词，只在片段内部取二元组。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class RagError(ValueError):
    """切块参数或索引配置不合法。"""


@dataclass(frozen=True)
class Chunk:
    """一个可检索的最小单元。"""

    doc_id: str
    chunk_index: int
    text: str
    vector: tuple[float, ...]

    @property
    def ref(self) -> str:
        """引用标识，形如 refund.md#2。"""
        return f"{self.doc_id}#{self.chunk_index}"


@dataclass(frozen=True)
class Citation:
    """检索命中的一条证据。"""

    doc_id: str
    chunk_index: int
    score: float
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"


@dataclass(frozen=True)
class SearchResult:
    """检索结果。`found=False` 是**正常结果**，不是错误。"""

    query: str
    found: bool
    citations: tuple[Citation, ...]
    reason: str | None = None


# --------------------------------------------------------------------- 切块


def split_text(text: str, *, max_chars: int = 220, overlap: int = 40) -> list[str]:
    """按段落切块，超长段落再硬切（带重叠）。

    为什么要重叠：一句话被切断在两块里时，重叠能让至少一块保留完整语义。
    """
    if max_chars <= 0:
        raise RagError("max_chars 必须大于 0")
    if overlap < 0 or overlap >= max_chars:
        raise RagError("overlap 必须满足 0 <= overlap < max_chars")

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    buffer = ""

    for block in blocks:
        # 标题是天然的语义边界：遇到 "#" 开头的块就先收口，
        # 避免把"配送时效"和"运费规则"揉进同一块（那会让检索变糊）。
        if block.startswith("#") and buffer:
            chunks.append(buffer)
            buffer = ""
        if len(block) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            step = max_chars - overlap
            for start in range(0, len(block), step):
                piece = block[start : start + max_chars]
                if piece.strip():
                    chunks.append(piece)
                if start + max_chars >= len(block):
                    break
            continue

        if buffer and len(buffer) + len(block) + 2 > max_chars:
            chunks.append(buffer)
            buffer = block
        else:
            buffer = f"{buffer}\n\n{block}" if buffer else block

    if buffer:
        chunks.append(buffer)
    return chunks


# --------------------------------------------------------------------- 分词


def _tokenize(text: str) -> list[str]:
    """按"片段"分词：字母数字按词，汉字片段内部取二元组。

    只在片段内部取 bigram，避免跨标点产生 "送与" 这类噪声。
    """
    lowered = text.lower()
    tokens: list[str] = []
    for segment in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", lowered):
        if segment.isascii():
            tokens.append(segment)
            continue
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return tokens


def build_idf(token_lists: Sequence[Sequence[str]]) -> dict[str, float]:
    """语料级 IDF：越少见的词权重越高。

    没有它的话，"的"、"商品"这种到处出现的词会把所有文档拉得一样相似，
    真正有区分度的词（包邮、保修）反而被淹没。
    """
    total = len(token_lists)
    if total == 0:
        return {}
    document_frequency: Counter[str] = Counter()
    for tokens in token_lists:
        document_frequency.update(set(tokens))
    return {
        token: math.log(1.0 + total / df)
        for token, df in document_frequency.items()
    }


# --------------------------------------------------------------------- 向量化


def embed(
    text: str,
    *,
    dim: int = 2048,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """确定性哈希向量化（feature hashing）+ 可选 IDF 权重。

    同一段文本 → 同一组浮点数；不依赖任何模型或随机种子。
    """
    if dim <= 0:
        raise RagError("dim 必须大于 0")

    vector = [0.0] * dim
    for token, count in Counter(_tokenize(text)).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = weights.get(token, 1.0) if weights else 1.0
        vector[index] += sign * weight * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度（两个向量都已归一化时等价于点积）。"""
    if len(left) != len(right):
        raise RagError("向量维度不一致")
    return sum(a * b for a, b in zip(left, right, strict=True))


# --------------------------------------------------------------------- 索引


class PolicyIndex:
    """文档索引。构建一次，反复查询。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        dim: int = 2048,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._chunks = tuple(chunks)
        self._dim = dim
        self._weights = dict(weights or {})

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._chunks

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[tuple[str, str]],
        *,
        max_chars: int = 220,
        overlap: int = 40,
        dim: int = 2048,
    ) -> PolicyIndex:
        # 第一遍：切块（先不向量化，因为 IDF 需要看全语料）
        raw: list[tuple[str, int, str]] = []
        for doc_id, text in documents:
            for index, piece in enumerate(
                split_text(text, max_chars=max_chars, overlap=overlap)
            ):
                raw.append((doc_id, index, piece))

        # 过滤过短的块（例如只有一行标题）：它们几乎没有信息量，
        # 却因为"短"而在余弦相似度里占便宜，会挤掉真正有内容的段落。
        raw = [item for item in raw if len(item[2].strip()) >= 20]

        # 第二遍：算 IDF，再向量化
        weights = build_idf([_tokenize(piece) for _, _, piece in raw])
        chunks = [
            Chunk(
                doc_id=doc_id,
                chunk_index=index,
                text=piece,
                vector=embed(piece, dim=dim, weights=weights),
            )
            for doc_id, index, piece in raw
        ]
        return cls(chunks, dim=dim, weights=weights)

    @classmethod
    def from_directory(
        cls,
        path: str | Path = "bootcamp/data/docs",
        *,
        max_chars: int = 220,
        overlap: int = 40,
        dim: int = 2048,
    ) -> PolicyIndex:
        directory = Path(path)
        if not directory.exists():
            raise RagError(f"文档目录不存在: {directory}")
        documents = [
            (file.name, file.read_text(encoding="utf-8"))
            for file in sorted(directory.glob("*.md"))
        ]
        if not documents:
            raise RagError(f"目录里没有 .md 文档: {directory}")
        return cls.from_documents(
            documents, max_chars=max_chars, overlap=overlap, dim=dim
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.12,
    ) -> SearchResult:
        """检索 top-k 证据。**低于阈值就不返回**，宁可说"没找到"。"""
        if not query.strip():
            return SearchResult(query=query, found=False, citations=(), reason="查询为空")
        if top_k <= 0:
            raise RagError("top_k 必须大于 0")

        query_vector = embed(query, dim=self._dim, weights=self._weights)
        scored = [(chunk, cosine(query_vector, chunk.vector)) for chunk in self._chunks]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        best = scored[0][1] if scored else 0.0
        hits = [
            Citation(
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
                score=round(score, 4),
                text=chunk.text,
            )
            for chunk, score in scored[:top_k]
            if score >= min_score
        ]

        if not hits:
            return SearchResult(
                query=query,
                found=False,
                citations=(),
                reason=f"没有条款达到相关度阈值 {min_score}（最高 {best:.4f}）",
            )
        return SearchResult(query=query, found=True, citations=tuple(hits))

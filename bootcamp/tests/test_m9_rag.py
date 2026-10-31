"""M9 验收测试：切块、向量化、检索、拒答、引用。

运行：uv run pytest bootcamp/tests/test_m9_rag.py -q

测试的重点不是"能查到"，而是三件事：
    1. 确定性（同输入同输出，可回放）
    2. 拒答（不相关就说不相关，而不是硬凑）
    3. 可追溯（每条结果能定位到原文）
"""

import pytest

from bootcamp.day2.m9_rag import (
    PolicyIndex,
    RagError,
    build_idf,
    cosine,
    embed,
    split_text,
)

DOCS = "bootcamp/data/docs"


@pytest.fixture(scope="module")
def index() -> PolicyIndex:
    return PolicyIndex.from_directory(DOCS)


# ------------------------------------------------------------------ 切块


def test_split_respects_max_chars() -> None:
    text = "\n\n".join(["段落" * 40] * 3)

    pieces = split_text(text, max_chars=120, overlap=20)

    assert all(len(p) <= 120 for p in pieces)
    assert len(pieces) > 1


def test_split_has_overlap_on_long_block() -> None:
    text = "甲" * 300

    pieces = split_text(text, max_chars=100, overlap=30)

    assert len(pieces) >= 3
    # 相邻块之间至少有 20 个字符重叠（重叠确保语义不被切断）
    assert pieces[0][-20:] in pieces[1]


def test_split_breaks_on_markdown_heading() -> None:
    text = "# 标题\n\n第一段内容\n\n## 小节\n\n第二段内容"

    pieces = split_text(text, max_chars=200)

    assert len(pieces) == 2  # 标题是硬边界
    assert pieces[1].startswith("## 小节")


@pytest.mark.parametrize(
    ("max_chars", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)]
)
def test_split_rejects_bad_params(max_chars: int, overlap: int) -> None:
    with pytest.raises(RagError):
        split_text("x", max_chars=max_chars, overlap=overlap)


# ------------------------------------------------------------------ 向量化


def test_embed_is_deterministic() -> None:
    assert embed("七天无理由退货") == embed("七天无理由退货")


def test_embed_is_normalized() -> None:
    vector = embed("保修期限为 12 个月")

    norm = sum(v * v for v in vector) ** 0.5

    assert norm == pytest.approx(1.0)


def test_embed_differs_for_different_text() -> None:
    assert embed("退货政策") != embed("运费规则")


def test_cosine_of_identical_is_one() -> None:
    vector = embed("预留 30 分钟")

    assert cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_rejects_dimension_mismatch() -> None:
    with pytest.raises(RagError):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_idf_gives_rare_tokens_higher_weight() -> None:
    corpus = [["包邮", "商品"], ["商品", "退货"], ["商品", "保修"]]

    idf = build_idf(corpus)

    assert idf["包邮"] > idf["商品"]


# ------------------------------------------------------------------ 检索命中


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("退货要几天内申请", "refund.md"),
        ("质量问题退货运费谁承担", "refund.md"),
        ("满多少包邮", "shipping.md"),
        ("新疆要加运费吗", "shipping.md"),
        ("保修多久", "warranty.md"),
        ("进水了还能保修吗", "warranty.md"),
        ("为什么可售数量比在库数量少", "inventory_faq.md"),
        ("预留多久会自动释放", "inventory_faq.md"),
    ],
)
def test_search_hits_expected_document(
    index: PolicyIndex, query: str, expected_doc: str
) -> None:
    result = index.search(query, top_k=1)

    assert result.found is True
    assert result.citations[0].doc_id == expected_doc


def test_search_returns_citations_with_ref(index: PolicyIndex) -> None:
    result = index.search("退货要几天", top_k=2)

    citation = result.citations[0]
    assert citation.doc_id == "refund.md"
    assert citation.chunk_index >= 0
    assert citation.ref == f"{citation.doc_id}#{citation.chunk_index}"
    assert "7 天" in citation.text  # 命中块里确实有答案


def test_top_k_limits_results(index: PolicyIndex) -> None:
    result = index.search("商品", top_k=2)

    assert len(result.citations) <= 2


def test_results_sorted_by_score(index: PolicyIndex) -> None:
    result = index.search("退货 运费 保修", top_k=3)

    scores = [c.score for c in result.citations]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ 拒答


@pytest.mark.parametrize("query", ["今天天气怎么样", "怎么开发票", "帮我写一首诗"])
def test_unrelated_query_is_rejected(index: PolicyIndex, query: str) -> None:
    result = index.search(query)

    assert result.found is False
    assert result.citations == ()
    assert "阈值" in (result.reason or "")


def test_empty_query_is_rejected(index: PolicyIndex) -> None:
    result = index.search("   ")

    assert result.found is False
    assert result.reason == "查询为空"


def test_known_limitation_lexical_false_positive(index: PolicyIndex) -> None:
    """已知局限：词面匹配分不清"支持"和"不支持"。

    "你们支持比特币支付吗" 会命中退款条款（因为里面有"不支持退换"和"支付"）。
    这不是 bug，而是 lexical retrieval 的天花板 —— 生产环境要靠 embedding
    模型或 cross-encoder 重排来解决。这里显式记录，避免以后误以为它是对的。
    """
    result = index.search("你们支持比特币支付吗")

    assert result.found is True                      # 当前行为：误命中
    assert result.citations[0].doc_id != "shipping.md"

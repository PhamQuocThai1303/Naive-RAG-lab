"""
selftest.py — kiểm chứng lõi RAG Lab, không cần GUI.

    python selftest.py

"""

from __future__ import annotations

import os
import sys
from typing import List

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

import rag_core as R

FAILS: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ==========================================================================


def test_chunking() -> None:
    print("\n1. Chunking (Cắt chunk)")
    text = "một hai ba bốn năm " * 40  # 760 ký tự

    cs = R.chunk_fixed(text, size=200, overlap=50)
    check("cửa sổ cố định phủ hết văn bản", cs[-1].end >= len(text) - 1,
          f"end={cs[-1].end}, len={len(text)}")
    check("mọi chunk không vượt quá size", all(c.n_chars <= 200 for c in cs))
    steps = [cs[i + 1].start - cs[i].start for i in range(len(cs) - 1)]
    check("bước nhảy = size - overlap", all(s == 150 for s in steps), str(set(steps)))

    no_ov = R.chunk_fixed(text, size=200, overlap=0)
    check("overlap làm tăng số chunk", len(cs) > len(no_ov),
          f"{len(cs)} vs {len(no_ov)}")
    check("không overlap thì tổng ký tự ≈ văn bản gốc",
          abs(sum(c.n_chars for c in no_ov) - len(text.strip())) < 20)

    doc = "Đoạn một.\n\nĐoạn hai dài hơn một chút.\n\nĐoạn ba."
    ps = R.chunk_paragraph(doc, size=500)
    check("cắt theo đoạn tách đúng 3 đoạn", len(ps) == 3, str(len(ps)))

    ss = R.chunk_sentence("Câu một. Câu hai! Câu ba? Câu bốn.", size=20)
    check("cắt theo câu không làm mất chữ", all(c.text.strip() for c in ss))

    check("size <= 0 phải báo lỗi", _raises(lambda: R.chunk_fixed("abc", size=0)))


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def test_merge_short() -> None:
    print("\n2. Merge short chunks (Bẫy orphan heading)")
    docs = [("t.txt", R.SAMPLE_DOC)]

    raw = R.chunk_documents(docs, "By Paragraph", min_chars=0)
    merged = R.chunk_documents(docs, "By Paragraph", min_chars=80)
    check("có chunk siêu ngắn khi không gộp", any(c.n_chars < 30 for c in raw))
    check("gộp xong không còn chunk siêu ngắn",
          all(c.n_chars >= 60 for c in merged),
          str(min(c.n_chars for c in merged)))
    check("gộp làm giảm số chunk", len(merged) < len(raw), f"{len(merged)} vs {len(raw)}")

    total_raw = sum(len(c.text) for c in raw)
    total_mg = sum(len(c.text) for c in merged)
    check("gộp không làm mất nội dung", total_mg >= total_raw - len(raw),
          f"{total_mg} vs {total_raw}")

    # chính là hiện tượng cần dạy: chunk ngắn ăn điểm cao giả tạo
    p_raw = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=0, top_k=1)
    p_raw.index(docs)
    top_raw = p_raw.ask("RAG là gì?").hits[0]

    p_mg = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=1)
    p_mg.index(docs)
    top_mg = p_mg.ask("RAG là gì?").hits[0]

    check("không gộp -> hạng 1 là mẩu tiêu đề rỗng nghĩa",
          top_raw.chunk.n_chars < 30 and top_raw.score > 0.9,
          f"{top_raw.chunk.n_chars} ký tự, điểm {top_raw.score:.3f}")
    check("gộp rồi -> hạng 1 là đoạn có nội dung thật",
          top_mg.chunk.n_chars > 100 and "Retrieval-Augmented" in top_mg.chunk.text,
          f"{top_mg.chunk.n_chars} ký tự")


def test_embedding_math() -> None:
    print("\n3. Tính chất toán học của embedding (L2 normalization, Cosine similarity)")
    texts = ["cosine similarity đo góc giữa hai vector",
             "chunk quá nhỏ thì thiếu ngữ cảnh",
             "RAG là retrieval augmented generation"]

    for name in ("TF-IDF", "Hashing"):
        emb = R.EMBEDDERS[name]()
        emb.fit(texts)
        m = emb.encode(texts)
        norms = np.linalg.norm(m, axis=1)
        check(f"{name}: mọi vector có độ dài 1",
              np.allclose(norms, 1.0, atol=1e-5), str(norms))
        check(f"{name}: tự so với chính mình = 1.0",
              abs(float(m[0] @ m[0]) - 1.0) < 1e-5)
        check(f"{name}: cosine luôn nằm trong [-1, 1]",
              bool(np.all(np.abs(m @ m.T) <= 1.0 + 1e-5)))

    emb = R.TfidfEmbedder()
    emb.fit(texts)
    same = emb.encode(["cosine similarity đo góc giữa hai vector"])[0]
    check("văn bản giống hệt -> cosine = 1",
          abs(float(same @ emb.encode(texts)[0]) - 1.0) < 1e-5)

    # đây là giới hạn CỐT LÕI của TF-IDF, cần khẳng định rõ chứ không giấu
    e2 = R.TfidfEmbedder()
    e2.fit(["tôi mua một chiếc xe hơi", "tôi mua một chiếc ô tô"])
    v = e2.encode(["xe hơi", "ô tô"])
    check("TF-IDF: từ đồng nghĩa khác mặt chữ -> cosine = 0",
          abs(float(v[0] @ v[1])) < 1e-6, f"{float(v[0] @ v[1]):.4f}")

    check("encode trước fit phải báo lỗi",
          _raises(lambda: R.TfidfEmbedder().encode(["x"])))

    terms = emb.top_terms("cosine similarity đo góc", 4)
    check("top_terms trả về từ có thật trong câu",
          all(w in "cosine similarity đo góc" for w, _ in terms), str(terms))


def test_retrieval() -> None:
    print("\n4. Retrieval (Truy hồi)")
    p = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=3)
    info = p.index(R.sample_docs())
    check("lập chỉ mục sinh ra chunk", info["n_chunks"] > 3, str(info["n_chunks"]))
    check("số chiều > 0", info["dim"] > 0)

    r = p.ask("cosine similarity đo cái gì?")
    check("trả về đúng top_k", len(r.hits) == 3, str(len(r.hits)))
    check("điểm giảm dần theo hạng",
          all(r.hits[i].score >= r.hits[i + 1].score for i in range(len(r.hits) - 1)))
    check("all_scores phủ mọi chunk",
          len(r.all_scores) == info["n_chunks"], str(len(r.all_scores)))
    check("hạng 1 đúng là đoạn nói về cosine",
          "cosine" in r.hits[0].chunk.text.lower(), r.hits[0].chunk.preview(50))
    check("rank đánh số từ 1", [h.rank for h in r.hits] == [1, 2, 3])

    top_manual = int(np.argmax(r.all_scores))
    check("hạng 1 khớp với argmax của all_scores",
          r.hits[0].chunk.id == top_manual)

    # tính chất quan trọng: naive RAG KHÔNG biết khi nào nó trượt
    off = p.ask("cách nấu phở bò truyền thống")
    check("câu hỏi lạc đề -> điểm cao nhất vẫn thấp",
          off.all_scores.max() < 0.2, f"{off.all_scores.max():.3f}")
    check("nhưng vẫn trả về đủ k đoạn (nó không tự biết đang trượt)",
          len(off.hits) == 3 or off.all_scores.max() < p.min_score,
          str(len(off.hits)))

    on = p.ask("cosine similarity")
    check("câu đúng chủ đề ăn điểm cao hơn hẳn câu lạc đề",
          on.all_scores.max() > off.all_scores.max() * 2,
          f"{on.all_scores.max():.3f} vs {off.all_scores.max():.3f}")

    p.min_score = 0.9
    check("ngưỡng cao thì lọc bớt kết quả", len(p.ask("chunk").hits) <= 1)
    p.min_score = 0.0

    st = r.stats()
    check("stats có đủ khoá", ({"max", "mean", "min"} <= set(st)) or ({"cao nhất", "trung bình", "thấp nhất"} <= set(st)))
    check("margin không âm", r.margin() >= 0)

    check("hỏi khi chưa lập chỉ mục -> báo lỗi",
          _raises(lambda: R.Pipeline().ask("x")))


def test_topk_effect() -> None:
    print("\n5. Ảnh hưởng của Top-k (Top-k effect)")
    p = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=1)
    p.index(R.sample_docs())
    q = "vì sao phải cắt chunk chồng lấn"

    got = {}
    for k in (1, 3, 6):
        p.top_k = k
        r = p.ask(q)
        got[k] = [h.chunk.id for h in r.hits]

    check("k lớn hơn thì lấy nhiều đoạn hơn",
          len(got[1]) < len(got[3]) < len(got[6]), str({k: len(v) for k, v in got.items()}))
    check("k nhỏ là tập con của k lớn (thứ hạng ổn định)",
          got[1] == got[3][:1] and got[3] == got[6][:3])


def test_prompt() -> None:
    print("\n6. Build prompt")
    p = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=2)
    p.index(R.sample_docs())
    r = p.ask("cosine similarity là gì")
    system, user = R.build_prompt("cosine similarity là gì", r.hits)

    check("system nhắc mô hình chỉ dùng context", "ngữ cảnh" in system.lower() or "context" in system.lower())
    check("user chứa câu hỏi", "cosine similarity là gì" in user)
    check("user chứa nguyên văn đoạn hạng 1", r.hits[0].chunk.text[:40] in user)
    check("mỗi đoạn được đánh số để trích dẫn", "[1]" in user and "[2]" in user)
    check("có ghi kèm điểm số", f"{r.hits[0].score:.3f}" in user)

    _, empty = R.build_prompt("gì đó", [])
    check("không có đoạn nào vẫn dựng được prompt", "không truy hồi được" in empty or "không retrieve được" in empty)

    ans = R.answer_extractive(r)
    check("trả lời trích xuất nêu đúng số đoạn", "2 đoạn" in ans or "2 chunks" in ans, ans[:60])


def test_reproducible() -> None:
    print("\n7. Reproducibility & Edge cases (Tính lặp lại và các trường hợp biên)")
    docs = R.sample_docs()
    a = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=3)
    b = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=3)
    a.index(docs)
    b.index(docs)
    ra, rb = a.ask("cosine"), b.ask("cosine")
    check("cùng cấu hình -> cùng kết quả",
          [h.chunk.id for h in ra.hits] == [h.chunk.id for h in rb.hits])
    check("cùng cấu hình -> cùng điểm",
          np.allclose(ra.all_scores, rb.all_scores))

    empty = R.chunk_documents([("rong.txt", "   \n\n  ")], "By Paragraph")
    check("tài liệu rỗng -> không sinh chunk nào", empty == [], str(empty))

    p = R.Pipeline(top_k=3)
    p.index([("x.txt", "chỉ một câu ngắn thôi.")])
    r = p.ask("câu ngắn")
    check("tài liệu 1 chunk vẫn chạy", len(r.hits) == 1, str(len(r.hits)))
    check("margin của 1 chunk = 0", r.margin() == 0.0)

    r2 = p.ask("")
    check("câu hỏi rỗng không làm sập", isinstance(r2.all_scores, np.ndarray))

    long_q = "chunk " * 500
    check("câu hỏi rất dài không làm sập", len(p.ask(long_q).all_scores) == 1)

    check("normalize gộp dòng trống thừa",
          R.normalize("a\n\n\n\n\nb") == "a\n\nb")
    check("normalize gộp khoảng trắng", R.normalize("a     b") == "a b")


def test_viz() -> None:
    print("\n8. Visualization (Trực quan hoá pipeline)")
    import rag_viz as V

    p = R.Pipeline(chunk_strategy="By Paragraph", min_chunk_chars=80, top_k=4)
    p.index(R.sample_docs())
    r = p.ask("cosine similarity đo cái gì")
    r._chunks = p.chunks

    d = V.PipelineDiagram(900, 96)
    d.set_step(2)
    check("sơ đồ pipeline dựng đúng kích thước", d.frame().size == (900, 96))

    c = V.ScoreChart(560, 300)
    check("biểu đồ rỗng vẫn dựng được", c.frame().size == (560, 300))
    c.set_result(r, 4, 0.05)
    img = c.frame()
    check("biểu đồ có điểm vẽ ra", _nonblank(img), "ảnh trống")
    check("biểu đồ giới hạn số thanh", len(c.scores) <= c.max_bars)

    m = V.EmbeddingMap(560, 300)
    m.set_data(p.store.matrix, r.query_vector, r.all_scores,
               [h.chunk.id for h in r.hits])
    check("bản đồ vector có nội dung", _nonblank(m.frame()))
    check("phương sai giữ lại nằm trong (0, 1]", 0 < m.var_kept <= 1.0,
          f"{m.var_kept:.3f}")

    m2 = V.EmbeddingMap(400, 200)
    m2.set_data(np.zeros((1, 5), dtype=np.float32), None, None, [])
    check("bản đồ với 1 điểm không làm sập", m2.frame().size == (400, 200))


def _nonblank(img, thresh: int = 800) -> bool:
    a = np.asarray(img.convert("RGB"), dtype=np.int32)
    bg = np.array([14, 17, 22])
    return int((np.abs(a - bg).sum(axis=2) > 40).sum()) > thresh


# ==========================================================================


def main() -> int:
    print("=" * 62)
    print("RAG Lab — Self-Test Suite")
    print("=" * 62)
    for fn in (test_chunking, test_merge_short, test_embedding_math,
               test_retrieval, test_topk_effect, test_prompt,
               test_reproducible, test_viz):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            FAILS.append(f"{fn.__name__}: {exc}")

    print("\n" + "=" * 62)
    if FAILS:
        print(f"FAILED (THẤT BẠI): {len(FAILS)}")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL TESTS PASSED (TẤT CẢ ĐỀU PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

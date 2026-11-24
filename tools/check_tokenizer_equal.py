import sys, time, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import meridianlm.tokenizer as T
from meridianlm.tokenizer import BPETokenizer
fast = T._fast
def both(docs, V):
    T._fast = None; t0=time.perf_counter(); py = BPETokenizer.train(docs, V); tp=time.perf_counter()-t0
    T._fast = fast; t0=time.perf_counter(); rs = BPETokenizer.train(docs, V); tr=time.perf_counter()-t0
    assert py.merges == rs.merges, "merges differ"
    assert py.merge_counts == rs.merge_counts, "counts differ"
    T._fast = None; pyt = BPETokenizer(py.merges)
    T._fast = fast; rst = BPETokenizer(py.merges)
    assert pyt._fast_enc is None and rst._fast_enc is not None
    text = "\n<|endoftext|>\n".join(docs)
    a, b = pyt.encode(text), rst.encode(text)
    assert a == b, "encodings differ"
    assert rst.decode(b) == text, "roundtrip"
    return tp, tr, len(py.merges), len(a)
docs = open(sys.argv[1] if len(sys.argv) > 1 else "corpus.txt", encoding="utf-8").read().split("\n<|endoftext|>\n")
for V in (300, 4096, 16384, 32768):
    tp, tr, m, n = both(docs, V)
    print(f"corpus V={V:>5}: identical ({m} merges, {n:,} tokens)  train python {tp:.2f}s rust {tr:.2f}s")
random.seed(1)
alphabet = list("aäöüßéèñ ząę\n\t  \r\x0b\x0c  漢字かな한국어😀👍🏽🇨🇭éä_-.,;:!?'\"()[]{}<>0123456789٣४") + ["'s","'ll"," the"," und"]
for trial in range(200):
    docs = ["".join(random.choice(alphabet) for _ in range(random.randint(0, 400))) for _ in range(random.randint(1, 12))]
    both(docs, random.choice([257, 260, 300, 600, 2000]))
print("200 random unicode corpora: identical")
for docs in ([""], ["a"], ["aaaa"*50], ["ab"*100, "ba"*100], ["<|endoftext|>"*3], [" "*50 + "\n"*20]):
    both(docs, 400)
print("edge cases: identical")

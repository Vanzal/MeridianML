import sys, time, statistics, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import meridianlm.tokenizer as T
from meridianlm.tokenizer import BPETokenizer
FAST = T._fast
docs = open(sys.argv[1] if len(sys.argv) > 1 else "corpus.txt", encoding="utf-8").read().split("\n<|endoftext|>\n")
text = "\n<|endoftext|>\n".join(docs)
merges = BPETokenizer.train(docs, 4096).merges
def med(f, n=6):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); xs.append(time.perf_counter() - t0)
    return statistics.median(xs), xs
def metrics(use_rust):
    T._fast = FAST if use_rust else None
    r = {}
    r["train4096_s"], _ = med(lambda: BPETokenizer.train(docs, 4096))
    r["train16384_s"], _ = med(lambda: BPETokenizer.train(docs, 16384))
    r["encode_cold_s"], _ = med(lambda: BPETokenizer(merges).encode(text))     # neuer Tokenizer = leerer Cache
    tok = BPETokenizer(merges); tok.encode(text)
    r["encode_warm_s"], _ = med(lambda: tok.encode(text))
    return {k: round(v, 3) for k, v in r.items()}
A = metrics(False); B = metrics(True); C = metrics(False)
print("control A ", A); print("candidate ", B); print("control B ", C)
out = {}
for k in A:
    worst = max(A[k], C[k]); best = min(A[k], C[k])
    out[k] = {"python_median_s": round((A[k]+C[k])/2, 3), "rust_median_s": B[k],
              "speedup_vs_better_control": round(best / B[k], 2), "beats_both": B[k] < best}
print(json.dumps(out, indent=1))
json.dump({"control_a": A, "candidate": B, "control_b": C, "summary": out,
           "corpus": "16 MB mixed (Python stdlib code + license prose + German README), 1203 docs",
           "machine": "2-core cloud VM, Python 3.13"}, open("tokenizer-ab.json", "w"), indent=1)

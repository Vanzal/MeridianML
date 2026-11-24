//! Rust-Kern für den BPE-Tokenizer von Meridian.
//!
//! Macht genau dasselbe wie die Python-Version in `meridianlm/tokenizer.py` – nur schneller:
//!
//! * `train`: Paare zählen und immer das häufigste verschmelzen. Bei Gleichstand gewinnt das kleinere Paar
//!   `(a, b)` – exakt die Reihenfolge des Python-Heaps `(-count, a, b)`.
//! * `Encoder.encode_words`: jedes Wort einzeln zerlegen; es wird immer das Paar mit dem kleinsten Rang verschmolzen
//!   (bei mehreren gleich: das linkeste), eins nach dem anderen.
//!
//! Die Vor-Zerlegung in Wörter (Regex) bleibt in Python, weil Pythons `\w`/`\s` Unicode etwas anders einteilt als
//! Rusts Regex-Bibliothek – so sind die Ergebnisse garantiert identisch.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Mutex;

type Pair = (u32, u32);

/// BPE-Training. `words`: die verschiedenen Wörter als UTF-8-Bytes, `freqs`: wie oft jedes vorkommt.
/// Gibt `(merges, merge_counts)` zurück. `progress(m, n, token_bytes)` wird alle 25 Merges aufgerufen.
#[pyfunction]
#[pyo3(signature = (words, freqs, n_merges, progress=None))]
fn train(
    py: Python<'_>,
    words: Vec<Vec<u8>>,
    freqs: Vec<i64>,
    n_merges: usize,
    progress: Option<PyObject>,
) -> PyResult<(Vec<Pair>, Vec<i64>)> {
    if words.len() != freqs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "words und freqs sind verschieden lang",
        ));
    }
    let mut words: Vec<Vec<u32>> = words
        .into_iter()
        .map(|w| w.into_iter().map(u32::from).collect())
        .collect();

    let mut pair_counts: HashMap<Pair, i64> = HashMap::new();
    let mut pair_words: HashMap<Pair, HashSet<usize>> = HashMap::new();
    for (i, w) in words.iter().enumerate() {
        for p in w.windows(2) {
            let pair = (p[0], p[1]);
            *pair_counts.entry(pair).or_insert(0) += freqs[i];
            pair_words.entry(pair).or_default().insert(i);
        }
    }
    // Max-Heap über (count, kleinstes a, kleinstes b) = Python-Min-Heap über (-count, a, b)
    let mut heap: BinaryHeap<(i64, Reverse<u32>, Reverse<u32>)> = pair_counts
        .iter()
        .map(|(&(a, b), &c)| (c, Reverse(a), Reverse(b)))
        .collect();

    let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut merges: Vec<Pair> = Vec::with_capacity(n_merges);
    let mut counts: Vec<i64> = Vec::with_capacity(n_merges);

    let result: PyResult<()> = py.allow_threads(|| {
        for m in 0..n_merges {
            let mut best: Option<Pair> = None;
            while let Some((neg, Reverse(a), Reverse(b))) = heap.pop() {
                let cur = *pair_counts.get(&(a, b)).unwrap_or(&0);
                if cur == neg {
                    best = Some((a, b));
                    break;
                }
                if cur > 0 {
                    heap.push((cur, Reverse(a), Reverse(b)));
                }
            }
            let best = match best {
                Some(p) if pair_counts.get(&p).copied().unwrap_or(0) >= 2 => p,
                _ => break,
            };
            let new_id = 256 + m as u32;
            counts.push(pair_counts[&best]);
            merges.push(best);
            let mut tok = vocab[best.0 as usize].clone();
            tok.extend_from_slice(&vocab[best.1 as usize]);
            vocab.push(tok);

            let mut touched: HashSet<Pair> = HashSet::new();
            let mut members: Vec<usize> = pair_words
                .get(&best)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            members.sort_unstable();
            for wi in members {
                let f = freqs[wi];
                let w = &words[wi];
                for p in w.windows(2) {
                    let pair = (p[0], p[1]);
                    if let Some(c) = pair_counts.get_mut(&pair) {
                        *c -= f;
                        if *c <= 0 {
                            pair_counts.remove(&pair);
                        }
                    }
                }
                let mut new_w: Vec<u32> = Vec::with_capacity(w.len());
                let mut i = 0;
                while i < w.len() {
                    if i + 1 < w.len() && w[i] == best.0 && w[i + 1] == best.1 {
                        new_w.push(new_id);
                        i += 2;
                    } else {
                        new_w.push(w[i]);
                        i += 1;
                    }
                }
                for p in new_w.windows(2) {
                    let pair = (p[0], p[1]);
                    *pair_counts.entry(pair).or_insert(0) += f;
                    pair_words.entry(pair).or_default().insert(wi);
                    touched.insert(pair);
                }
                words[wi] = new_w;
            }
            pair_words.remove(&best);
            pair_counts.remove(&best);
            for p in touched {
                if let Some(&c) = pair_counts.get(&p) {
                    if c != 0 {
                        heap.push((c, Reverse(p.0), Reverse(p.1)));
                    }
                }
            }
            if let Some(cb) = &progress {
                if m % 25 == 0 || m + 1 == n_merges {
                    let last = vocab.last().cloned().unwrap_or_default();
                    Python::with_gil(|py| -> PyResult<()> {
                        cb.call1(py, (m + 1, n_merges, PyBytes::new(py, &last)))?;
                        Ok(())
                    })?;
                }
            }
        }
        Ok(())
    });
    result?;
    Ok((merges, counts))
}

/// Zerlegt Wörter in Token-IDs anhand fester Merges. Merkt sich bereits zerlegte Wörter (wie der Python-Cache).
#[pyclass(frozen)]
struct Encoder {
    rank: HashMap<Pair, u32>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

impl Encoder {
    fn encode_word(&self, word: &str) -> Vec<u32> {
        if let Some(hit) = self.cache.lock().unwrap().get(word) {
            return hit.clone();
        }
        let mut ids: Vec<u32> = word.bytes().map(u32::from).collect();
        while ids.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..ids.len() - 1 {
                if let Some(&r) = self.rank.get(&(ids[i], ids[i + 1])) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            match best {
                None => break,
                Some((r, i)) => {
                    ids[i] = 256 + r;
                    ids.remove(i + 1);
                }
            }
        }
        let mut cache = self.cache.lock().unwrap();
        if cache.len() < 500_000 {
            cache.insert(word.to_owned(), ids.clone());
        }
        ids
    }
}

#[pymethods]
impl Encoder {
    #[new]
    fn new(merges: Vec<Pair>) -> Self {
        let rank = merges
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p, i as u32))
            .collect();
        Encoder {
            rank,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Alle Wörter nacheinander zerlegen und die IDs aneinanderhängen.
    fn encode_words(&self, py: Python<'_>, words: Vec<String>) -> Vec<u32> {
        py.allow_threads(|| {
            let mut out = Vec::with_capacity(words.len() * 2);
            for w in &words {
                out.extend(self.encode_word(w));
            }
            out
        })
    }

    fn encode_one(&self, word: &str) -> Vec<u32> {
        self.encode_word(word)
    }
}

#[pymodule]
fn meridian_fastbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_class::<Encoder>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

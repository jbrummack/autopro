//! Native SentencePiece: reads `.model` files and encodes exactly like the
//! `sentencepiece` library (normalizer, unigram Viterbi, BPE merges), without
//! the C++ dependency.
use std::collections::{BinaryHeap, HashMap};

use prost::Message;

use crate::error::{Error, Result};

// --- The parts of sentencepiece_model.proto that encoding needs.

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<PieceProto>,
    #[prost(message, optional, tag = "2")]
    trainer_spec: Option<TrainerSpec>,
    #[prost(message, optional, tag = "3")]
    normalizer_spec: Option<NormalizerSpec>,
}

#[derive(Clone, PartialEq, Message)]
struct PieceProto {
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    #[prost(float, optional, tag = "2")]
    score: Option<f32>,
    #[prost(int32, optional, tag = "3")]
    kind: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct TrainerSpec {
    #[prost(int32, optional, tag = "3")]
    model_type: Option<i32>,
    #[prost(bool, optional, tag = "24")]
    treat_whitespace_as_suffix: Option<bool>,
    #[prost(bool, optional, tag = "35")]
    byte_fallback: Option<bool>,
    #[prost(int32, optional, tag = "40")]
    unk_id: Option<i32>,
    #[prost(int32, optional, tag = "41")]
    bos_id: Option<i32>,
    #[prost(int32, optional, tag = "42")]
    eos_id: Option<i32>,
    #[prost(int32, optional, tag = "43")]
    pad_id: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct NormalizerSpec {
    #[prost(string, optional, tag = "1")]
    name: Option<String>,
    #[prost(bytes = "vec", optional, tag = "2")]
    precompiled_charsmap: Option<Vec<u8>>,
    #[prost(bool, optional, tag = "3")]
    add_dummy_prefix: Option<bool>,
    #[prost(bool, optional, tag = "4")]
    remove_extra_whitespaces: Option<bool>,
    #[prost(bool, optional, tag = "5")]
    escape_whitespaces: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceKind {
    fn from_proto(kind: Option<i32>) -> PieceKind {
        match kind.unwrap_or(1) {
            2 => PieceKind::Unknown,
            3 => PieceKind::Control,
            4 => PieceKind::UserDefined,
            5 => PieceKind::Unused,
            6 => PieceKind::Byte,
            _ => PieceKind::Normal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Unigram,
    Bpe,
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    score: f32,
    kind: PieceKind,
}

const SPACE_SYMBOL: &str = "\u{2581}";

// --- Normalizer: precompiled character map (a darts-clone double array).

#[derive(Debug, Clone, Default)]
struct CharsMap {
    trie: Vec<u32>,
    normalized: Vec<u8>,
}

impl CharsMap {
    fn parse(blob: &[u8]) -> Result<CharsMap> {
        if blob.is_empty() {
            return Ok(CharsMap::default());
        }
        let bad = || Error::Text("invalid precompiled_charsmap".into());
        let size = u32::from_le_bytes(blob.get(..4).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let trie_bytes = blob.get(4..4 + size).ok_or_else(bad)?;
        Ok(CharsMap {
            trie: trie_bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect(),
            normalized: blob[4 + size..].to_vec(),
        })
    }

    /// Longest rule matching a prefix of `input`: (replacement, consumed bytes).
    fn longest_prefix(&self, input: &[u8]) -> Option<(&[u8], usize)> {
        if self.trie.is_empty() {
            return None;
        }
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let label = |u: u32| u & ((1 << 31) | 0xFF);
        let has_leaf = |u: u32| (u >> 8) & 1 == 1;
        let mut node = offset(self.trie[0]);
        let mut best = None;
        for (i, &c) in input.iter().enumerate() {
            if c == 0 {
                break;
            }
            node ^= c as usize;
            let unit = *self.trie.get(node)?;
            if label(unit) != c as u32 {
                break;
            }
            node ^= offset(unit);
            if has_leaf(unit) {
                let value = (self.trie[node] & ((1 << 31) - 1)) as usize;
                let end = self.normalized[value..].iter().position(|&b| b == 0).map_or(self.normalized.len(), |p| value + p);
                best = Some((&self.normalized[value..end], i + 1));
            }
        }
        best
    }
}

/// A loaded SentencePiece model.
#[derive(Debug, Clone)]
pub struct SentencePiece {
    pieces: Vec<Piece>,
    ids: HashMap<String, u32>,
    /// Longest piece in bytes, bounding prefix lookups.
    max_piece_len: usize,
    model_type: ModelType,
    charsmap: CharsMap,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
    byte_fallback: bool,
    unk_id: u32,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
    min_score: f32,
    max_score: f32,
}

fn optional_id(id: i32) -> Option<u32> {
    (id >= 0).then_some(id as u32)
}

impl SentencePiece {
    pub fn from_bytes(bytes: &[u8]) -> Result<SentencePiece> {
        let proto = ModelProto::decode(bytes).map_err(|e| Error::Text(format!("not a SentencePiece model: {e}")))?;
        let trainer = proto.trainer_spec.unwrap_or_default();
        let normalizer = proto.normalizer_spec.unwrap_or_default();
        let model_type = match trainer.model_type.unwrap_or(1) {
            1 => ModelType::Unigram,
            2 => ModelType::Bpe,
            3 => return Err(Error::Text("SentencePiece WORD models are not supported".into())),
            4 => return Err(Error::Text("SentencePiece CHAR models are not supported".into())),
            t => return Err(Error::Text(format!("unknown SentencePiece model type {t}"))),
        };
        let pieces: Vec<Piece> = proto
            .pieces
            .iter()
            .map(|p| Piece {
                text: p.piece.clone().unwrap_or_default(),
                score: p.score.unwrap_or(0.0),
                kind: PieceKind::from_proto(p.kind),
            })
            .collect();
        let ids = pieces.iter().enumerate().map(|(i, p)| (p.text.clone(), i as u32)).collect();
        let normal_scores = pieces.iter().filter(|p| p.kind == PieceKind::Normal).map(|p| p.score);
        let (min_score, max_score) =
            normal_scores.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| (lo.min(s), hi.max(s)));
        Ok(SentencePiece {
            max_piece_len: pieces.iter().map(|p| p.text.len()).max().unwrap_or(0),
            ids,
            pieces,
            model_type,
            charsmap: CharsMap::parse(normalizer.precompiled_charsmap.as_deref().unwrap_or(&[]))?,
            add_dummy_prefix: normalizer.add_dummy_prefix.unwrap_or(true),
            remove_extra_whitespaces: normalizer.remove_extra_whitespaces.unwrap_or(true),
            escape_whitespaces: normalizer.escape_whitespaces.unwrap_or(true),
            treat_whitespace_as_suffix: trainer.treat_whitespace_as_suffix.unwrap_or(false),
            byte_fallback: trainer.byte_fallback.unwrap_or(false),
            unk_id: trainer.unk_id.unwrap_or(0).max(0) as u32,
            bos_id: optional_id(trainer.bos_id.unwrap_or(1)),
            eos_id: optional_id(trainer.eos_id.unwrap_or(2)),
            pad_id: optional_id(trainer.pad_id.unwrap_or(-1)),
            min_score,
            max_score,
        })
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    pub fn piece_to_id(&self, piece: &str) -> Option<u32> {
        self.ids.get(piece).copied()
    }

    pub fn id_to_piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.text.as_str())
    }

    fn kind(&self, id: u32) -> PieceKind {
        self.pieces[id as usize].kind
    }

    /// One normalization step: the longest charsmap rule at the start of
    /// `input`, or its first UTF-8 character unchanged.
    fn normalize_prefix<'a>(&'a self, input: &'a [u8]) -> (&'a [u8], usize) {
        if let Some(hit) = self.charsmap.longest_prefix(input) {
            return hit;
        }
        let len = match input[0] {
            b if b < 0x80 => 1,
            b if b >> 5 == 0b110 => 2,
            b if b >> 4 == 0b1110 => 3,
            _ => 4,
        }
        .min(input.len());
        (&input[..len], len)
    }

    /// SentencePiece's Normalizer::Normalize.
    pub fn normalize(&self, input: &str) -> String {
        let mut rest = input.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(input.len() + 3);
        let add_ws = |out: &mut Vec<u8>| {
            out.extend_from_slice(if self.escape_whitespaces { SPACE_SYMBOL.as_bytes() } else { b" " })
        };
        if self.remove_extra_whitespaces {
            while !rest.is_empty() {
                let (piece, len) = self.normalize_prefix(rest);
                if piece != b" " {
                    break;
                }
                rest = &rest[len..];
            }
        }
        if rest.is_empty() {
            return String::new();
        }
        if !self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            add_ws(&mut out);
        }
        let mut is_prev_space = self.remove_extra_whitespaces;
        while !rest.is_empty() {
            let (mut piece, len) = self.normalize_prefix(rest);
            while is_prev_space && piece.first() == Some(&b' ') {
                piece = &piece[1..];
            }
            if !piece.is_empty() {
                for &b in piece {
                    if self.escape_whitespaces && b == b' ' {
                        out.extend_from_slice(SPACE_SYMBOL.as_bytes());
                    } else {
                        out.push(b);
                    }
                }
                is_prev_space = piece.last() == Some(&b' ');
            }
            rest = &rest[len..];
            is_prev_space = is_prev_space && self.remove_extra_whitespaces;
        }
        if self.remove_extra_whitespaces {
            let space: &[u8] = if self.escape_whitespaces { SPACE_SYMBOL.as_bytes() } else { b" " };
            while out.ends_with(space) {
                out.truncate(out.len() - space.len());
            }
        }
        if self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            add_ws(&mut out);
        }
        // Rules map valid UTF-8 to valid UTF-8.
        String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    /// Pieces (by id) that are prefixes of `s`, shortest first.
    fn prefix_pieces<'a>(&'a self, s: &'a str) -> impl Iterator<Item = (usize, u32)> + 'a {
        s.char_indices()
            .map(|(i, c)| i + c.len_utf8())
            .take_while(|&end| end <= self.max_piece_len)
            .filter_map(|end| self.ids.get(&s[..end]).map(|&id| (end, id)))
    }

    /// Unigram Viterbi over the normalized text: (piece text, id) pairs.
    fn encode_unigram<'a>(&self, text: &'a str) -> Vec<(&'a str, u32)> {
        // Char boundaries: positions[k] is the byte offset of character k.
        let positions: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
        let n = positions.len() - 1;
        let char_index: HashMap<usize, usize> = positions.iter().enumerate().map(|(k, &b)| (b, k)).collect();
        let unk_score = self.min_score - 10.0;

        struct Node {
            begin: usize,
            end: usize,
            id: u32,
            score: f32,
            backtrace: f32,
            prev: usize,
        }
        // Node 0 is BOS; ends[k] lists nodes ending at char k in insertion order.
        let mut nodes = vec![Node { begin: 0, end: 0, id: 0, score: 0.0, backtrace: 0.0, prev: usize::MAX }];
        let mut begins: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        let mut ends: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        ends[0].push(0);
        for begin in 0..n {
            let start = positions[begin];
            let mut has_single = false;
            for (len, id) in self.prefix_pieces(&text[start..]) {
                let kind = self.kind(id);
                if !matches!(kind, PieceKind::Normal | PieceKind::UserDefined) {
                    continue;
                }
                let end = char_index[&(start + len)];
                let length = end - begin;
                let score = if kind == PieceKind::UserDefined {
                    length as f32 * self.max_score - 0.1
                } else {
                    self.pieces[id as usize].score
                };
                has_single |= length == 1;
                begins[begin].push(nodes.len());
                ends[end].push(nodes.len());
                nodes.push(Node { begin, end, id, score, backtrace: 0.0, prev: usize::MAX });
            }
            if !has_single {
                begins[begin].push(nodes.len());
                ends[begin + 1].push(nodes.len());
                nodes.push(Node { begin, end: begin + 1, id: self.unk_id, score: unk_score, backtrace: 0.0, prev: usize::MAX });
            }
        }
        for pos in 0..n {
            for &r in &begins[pos] {
                let mut best: Option<(usize, f32)> = None;
                for &l in &ends[pos] {
                    let score = nodes[l].backtrace + nodes[r].score;
                    if best.is_none_or(|(_, s)| score > s) {
                        best = Some((l, score));
                    }
                }
                if let Some((l, score)) = best {
                    nodes[r].prev = l;
                    nodes[r].backtrace = score;
                }
            }
        }
        // EOS: best node ending at n.
        let mut best: Option<(usize, f32)> = None;
        for &l in &ends[n] {
            if best.is_none_or(|(_, s)| nodes[l].backtrace > s) {
                best = Some((l, nodes[l].backtrace));
            }
        }
        let mut path = Vec::new();
        let mut node = best.map_or(usize::MAX, |(l, _)| l);
        while node != 0 && node != usize::MAX {
            let nd = &nodes[node];
            path.push((&text[positions[nd.begin]..positions[nd.end]], nd.id));
            node = nd.prev;
        }
        path.reverse();
        path
    }

    /// SentencePiece BPE: greedy merges of the highest-scoring adjacent pair.
    fn encode_bpe<'a>(&self, text: &'a str) -> Vec<(&'a str, u32)> {
        #[derive(Clone, Copy)]
        struct Symbol {
            start: usize,
            end: usize,
            prev: Option<usize>,
            next: Option<usize>,
            alive: bool,
        }
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut i = 0;
        while i < text.len() {
            // User-defined pieces stay whole.
            let user = self
                .prefix_pieces(&text[i..])
                .filter(|&(_, id)| self.kind(id) == PieceKind::UserDefined)
                .last()
                .map(|(len, _)| len);
            let len = user.unwrap_or_else(|| text[i..].chars().next().unwrap().len_utf8());
            let k = symbols.len();
            symbols.push(Symbol { start: i, end: i + len, prev: k.checked_sub(1), next: None, alive: true });
            if k > 0 {
                symbols[k - 1].next = Some(k);
            }
            i += len;
        }
        // Max-heap on score, ties to the leftmost pair.
        #[derive(PartialEq)]
        struct Pair {
            score: f32,
            left: usize,
            right: usize,
            len: usize,
        }
        impl Eq for Pair {}
        impl PartialOrd for Pair {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Pair {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score.total_cmp(&other.score).then(other.left.cmp(&self.left))
            }
        }
        let mut agenda = BinaryHeap::new();
        let try_pair = |agenda: &mut BinaryHeap<Pair>, symbols: &[Symbol], left: usize, right: usize| {
            let piece = &text[symbols[left].start..symbols[right].end];
            if let Some(&id) = self.ids.get(piece) {
                if matches!(self.kind(id), PieceKind::Normal | PieceKind::UserDefined | PieceKind::Unused) {
                    agenda.push(Pair { score: self.pieces[id as usize].score, left, right, len: piece.len() });
                }
            }
        };
        for k in 1..symbols.len() {
            try_pair(&mut agenda, &symbols, k - 1, k);
        }
        while let Some(pair) = agenda.pop() {
            let (l, r) = (pair.left, pair.right);
            // Skip pairs invalidated by earlier merges.
            if !symbols[l].alive || !symbols[r].alive || symbols[l].end - symbols[l].start + symbols[r].end - symbols[r].start != pair.len
            {
                continue;
            }
            symbols[l].end = symbols[r].end;
            symbols[l].next = symbols[r].next;
            if let Some(n) = symbols[r].next {
                symbols[n].prev = Some(l);
            }
            symbols[r].alive = false;
            if let Some(p) = symbols[l].prev {
                try_pair(&mut agenda, &symbols, p, l);
            }
            if let Some(n) = symbols[l].next {
                try_pair(&mut agenda, &symbols, l, n);
            }
        }
        let mut out = Vec::new();
        let mut k = if symbols.is_empty() { None } else { Some(0) };
        while let Some(s) = k {
            let piece = &text[symbols[s].start..symbols[s].end];
            let id = match self.ids.get(piece) {
                Some(&id) if matches!(self.kind(id), PieceKind::Normal | PieceKind::UserDefined) => id,
                _ => self.unk_id,
            };
            out.push((piece, id));
            k = symbols[s].next;
        }
        out
    }

    /// Token ids of `text`, like `SentencePieceProcessor.encode(text, add_bos, add_eos)`.
    /// Text from ids like the library's `DecodeIds`: pieces joined with "▁"
    /// as spaces, byte pieces assembled into UTF-8 (invalid bytes become
    /// U+FFFD), unknown ids as " ⁇ ", the dummy-prefix space removed.
    /// Control pieces (e.g. `<bos>`) are dropped, or kept as their text
    /// without `skip_special`. Ids outside the vocabulary are skipped.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |bytes: &mut Vec<u8>, out: &mut String| {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };
        let mut first_text = true;
        for &id in ids {
            let Some(piece) = self.pieces.get(id as usize) else { continue };
            if piece.kind == PieceKind::Byte {
                if let Some(b) = piece.text.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    bytes.push(b);
                    first_text = false;
                }
                continue;
            }
            flush(&mut bytes, &mut out);
            match piece.kind {
                PieceKind::Control if skip_special => {}
                PieceKind::Control => out.push_str(&piece.text),
                PieceKind::Unknown => {
                    out.push_str(" \u{2047} ");
                    first_text = false;
                }
                _ => {
                    let mut text = if self.escape_whitespaces { piece.text.replace('\u{2581}', " ") } else { piece.text.clone() };
                    if first_text && self.add_dummy_prefix && !self.treat_whitespace_as_suffix {
                        if let Some(rest) = text.strip_prefix(' ') {
                            text = rest.to_string();
                        }
                    }
                    first_text = false;
                    out.push_str(&text);
                }
            }
        }
        flush(&mut bytes, &mut out);
        out
    }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let normalized = self.normalize(text);
        let pieces = match self.model_type {
            ModelType::Unigram => self.encode_unigram(&normalized),
            ModelType::Bpe => self.encode_bpe(&normalized),
        };
        let mut ids = Vec::with_capacity(pieces.len() + 2);
        if add_bos {
            ids.extend(self.bos_id);
        }
        let mut prev_unk = false;
        for (piece, id) in pieces {
            let unk = id == self.unk_id;
            if unk && self.byte_fallback {
                for b in piece.bytes() {
                    ids.push(self.piece_to_id(&format!("<0x{b:02X}>")).unwrap_or(self.unk_id));
                }
            } else if !(unk && prev_unk) {
                // Runs of unknown pieces merge into one unknown token.
                ids.push(id);
            }
            prev_unk = unk && !self.byte_fallback;
        }
        if add_eos {
            ids.extend(self.eos_id);
        }
        ids
    }
}

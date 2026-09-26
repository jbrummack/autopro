//! Text to token ids.
pub mod sentencepiece;

use std::path::Path;

use ndarray::Array2;

use crate::error::{Error, Result};
pub use sentencepiece::SentencePiece;

/// A tokenizer: a SentencePiece `.model` or a Hugging Face `tokenizer.json`.
#[derive(Clone)]
pub enum Tokenizer {
    SentencePiece(SentencePiece),
    HuggingFace(Box<tokenizers::Tokenizer>),
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tokenizer::SentencePiece(sp) => write!(f, "SentencePiece({} pieces)", sp.vocab_size()),
            Tokenizer::HuggingFace(t) => write!(f, "HuggingFace({} tokens)", t.get_vocab_size(true)),
        }
    }
}

impl Tokenizer {
    /// Detects the format: JSON is a `tokenizer.json`, anything else a SentencePiece model.
    pub fn from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
        let first = bytes.iter().find(|b| !b.is_ascii_whitespace());
        if first == Some(&b'{') {
            let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| Error::Text(e.to_string()))?;
            Ok(Tokenizer::HuggingFace(Box::new(tokenizer)))
        } else {
            Ok(Tokenizer::SentencePiece(SentencePiece::from_bytes(bytes)?))
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Tokenizer> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Token ids. `add_special_tokens`: BOS/EOS for SentencePiece, the
    /// post-processor's tokens (e.g. [CLS]/[SEP]) for tokenizer.json.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match self {
            Tokenizer::SentencePiece(sp) => Ok(sp.encode(text, add_special_tokens, add_special_tokens)),
            Tokenizer::HuggingFace(t) => Ok(t
                .encode(text, add_special_tokens)
                .map_err(|e| Error::Text(e.to_string()))?
                .get_ids()
                .to_vec()),
        }
    }

    /// Like [`Tokenizer::encode`] but truncates the text's tokens so that the
    /// result, special tokens included, has at most `limit` ids (special
    /// tokens are kept, as in Hugging Face's truncation).
    pub fn encode_truncated(&self, text: &str, add_special_tokens: bool, limit: Option<usize>) -> Result<Vec<u32>> {
        let Some(limit) = limit else { return self.encode(text, add_special_tokens) };
        match self {
            Tokenizer::SentencePiece(sp) => {
                let (bos, eos) = if add_special_tokens { (sp.bos_id(), sp.eos_id()) } else { (None, None) };
                let mut ids = sp.encode(text, false, false);
                ids.truncate(limit.saturating_sub(bos.iter().len() + eos.iter().len()));
                Ok(bos.into_iter().chain(ids).chain(eos).collect())
            }
            Tokenizer::HuggingFace(t) => {
                let err = |e: tokenizers::Error| Error::Text(e.to_string());
                let mut encoding = t.encode(text, false).map_err(err)?;
                let special = match (add_special_tokens, t.get_post_processor()) {
                    (true, Some(p)) => tokenizers::PostProcessor::added_tokens(p, false),
                    _ => 0,
                };
                encoding.truncate(limit.saturating_sub(special), 0, tokenizers::TruncationDirection::Right);
                Ok(t.post_process(encoding, None, add_special_tokens).map_err(err)?.get_ids().to_vec())
            }
        }
    }

    /// Text from token ids; `skip_special` drops special tokens (`<bos>`, `<eos>`, ...).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        match self {
            Tokenizer::SentencePiece(sp) => Ok(sp.decode(ids, skip_special)),
            Tokenizer::HuggingFace(t) => t.decode(ids, skip_special).map_err(|e| Error::Text(e.to_string())),
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Tokenizer::SentencePiece(sp) => sp.piece_to_id(token),
            Tokenizer::HuggingFace(t) => t.token_to_id(token),
        }
    }

    /// The model's padding id, if it defines one.
    pub fn pad_id(&self) -> Option<u32> {
        match self {
            Tokenizer::SentencePiece(sp) => sp.pad_id(),
            Tokenizer::HuggingFace(t) => t.get_padding().map(|p| p.pad_id).or_else(|| {
                ["<pad>", "[PAD]", "<|pad|>", "<|endoftext|>"].iter().find_map(|p| t.token_to_id(p))
            }),
        }
    }
}

/// How sequences are padded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// No padding; batches must then have equal lengths.
    None,
    /// To the longest sequence of the batch.
    Longest,
    /// To exactly this length (after truncation to at most this length).
    Fixed(usize),
}

/// Token ids and attention mask (1 = token, 0 = padding), `[batch, length]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
}

/// Text preprocessing options: case, special tokens, truncation, padding.
#[derive(Debug, Clone, PartialEq)]
pub struct TextOptions {
    pub lowercase: bool,
    pub add_special_tokens: bool,
    /// Truncate to at most this many tokens (special tokens included).
    pub max_length: Option<usize>,
    pub padding: Padding,
    /// Id used for padding; defaults to the tokenizer's, else 0.
    pub pad_id: Option<u32>,
}

impl Default for TextOptions {
    fn default() -> Self {
        TextOptions { lowercase: false, add_special_tokens: true, max_length: None, padding: Padding::None, pad_id: None }
    }
}

impl TextOptions {
    /// Ids of one text, truncated but not padded.
    pub fn encode(&self, tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
        let text = if self.lowercase { text.to_lowercase() } else { text.to_string() };
        let limit = match self.padding {
            Padding::Fixed(n) => Some(self.max_length.map_or(n, |m| m.min(n))),
            _ => self.max_length,
        };
        tokenizer.encode_truncated(&text, self.add_special_tokens, limit)
    }

    pub fn encode_batch(&self, tokenizer: &Tokenizer, texts: &[&str]) -> Result<Encoded> {
        let encoded: Vec<Vec<u32>> = texts.iter().map(|t| self.encode(tokenizer, t)).collect::<Result<_>>()?;
        let longest = encoded.iter().map(Vec::len).max().unwrap_or(0);
        let length = match self.padding {
            Padding::Fixed(n) => n,
            Padding::Longest => longest,
            Padding::None => {
                if encoded.iter().any(|e| e.len() != longest) {
                    return Err(Error::Text("texts have different lengths; set padding".into()));
                }
                longest
            }
        };
        let pad = self.pad_id.or_else(|| tokenizer.pad_id()).unwrap_or(0) as i64;
        let mut ids = Array2::from_elem((texts.len(), length), pad);
        let mut mask = Array2::zeros((texts.len(), length));
        for (row, seq) in encoded.iter().enumerate() {
            for (col, &id) in seq.iter().enumerate() {
                ids[[row, col]] = id as i64;
                mask[[row, col]] = 1;
            }
        }
        Ok(Encoded { ids, attention_mask: mask })
    }
}

/// A tokenizer with its [`TextOptions`].
#[derive(Debug, Clone)]
pub struct TextProcessor {
    pub tokenizer: Tokenizer,
    pub options: TextOptions,
}

impl TextProcessor {
    pub fn new(tokenizer: Tokenizer) -> TextProcessor {
        TextProcessor { tokenizer, options: TextOptions::default() }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.options.encode(&self.tokenizer, text)
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Encoded> {
        self.options.encode_batch(&self.tokenizer, texts)
    }
}

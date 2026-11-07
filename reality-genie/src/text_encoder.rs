use std::error::Error;
use tokenizers::Tokenizer;

pub struct TextEncoder {
    tokenizer: Tokenizer,
}

impl TextEncoder {
    pub fn new(vocab_bytes: &[u8]) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_bytes(vocab_bytes)?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let encoding = self.tokenizer.encode(prompt, false)?;
        Ok(encoding.get_ids().to_vec())
    }
}

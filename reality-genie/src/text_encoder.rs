use tokenizers::Tokenizer;
use std::error::Error;

pub struct TextEncoder {
    tokenizer: Tokenizer,
}

impl TextEncoder {
    pub fn new(json_bytes: &[u8]) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_bytes(json_bytes)?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Ok(encoding) = self.tokenizer.encode(text, true) {
            encoding.get_ids().to_vec()
        } else {
            Vec::new()
        }
    }
}

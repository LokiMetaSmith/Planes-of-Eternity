// Test file to prototype the serialization format
use std::mem;

pub fn serialize_labels() -> Vec<u8> {
    let mut bytes = Vec::new();
    let count: u32 = 1;
    bytes.extend_from_slice(&count.to_le_bytes());

    let x: f32 = 0.5;
    let y: f32 = 0.5;
    let r: u8 = 255;
    let g: u8 = 255;
    let b: u8 = 0;
    let a: u8 = 255;
    let text = "TEST";

    bytes.extend_from_slice(&x.to_le_bytes());
    bytes.extend_from_slice(&y.to_le_bytes());
    bytes.push(r);
    bytes.push(g);
    bytes.push(b);
    bytes.push(a);

    let text_bytes = text.as_bytes();
    let len = text_bytes.len() as u16;
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(text_bytes);

    bytes
}

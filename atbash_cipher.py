import re

def encode(plain_text):
    plain_text = plain_text.lower()
    cleaned_text = re.sub(r'[^a-z0-9]', '', plain_text)

    encoded_chars = []
    for char in cleaned_text:
        if 'a' <= char <= 'z':
            encoded_chars.append(chr(219 - ord(char)))
        else:
            encoded_chars.append(char)

    result = []
    for i in range(0, len(encoded_chars), 5):
        result.append("".join(encoded_chars[i:i+5]))

    return " ".join(result)

def decode(ciphered_text):
    cleaned_text = re.sub(r'[^a-z0-9]', '', ciphered_text.lower())

    decoded_chars = []
    for char in cleaned_text:
        if 'a' <= char <= 'z':
            decoded_chars.append(chr(219 - ord(char)))
        else:
            decoded_chars.append(char)

    return "".join(decoded_chars)

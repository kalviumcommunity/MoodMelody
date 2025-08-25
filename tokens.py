from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "Gemini AI is powerful!"
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

token_ids = tokenizer.encode(sentence)
print("Token IDs:", token_ids)

decoded = tokenizer.decode(token_ids)
print("Decoded back:", decoded)

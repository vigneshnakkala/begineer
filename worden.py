import keras
from keras import ops
from keras import layers
import numpy as np

# Sample text data (replace with your own large text for better results)
text = "the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog"

# Tokenize words
words = text.split()
vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
maxlen = 5  # Sequence length

# Prepare input/output pairs
inputs = []
targets = []
for i in range(len(words) - maxlen):
    inputs.append([word2idx[w] for w in words[i:i+maxlen]])
    targets.append(word2idx[words[i+maxlen]])
inputs = np.array(inputs)
targets = np.array(targets)

# Model
embed_dim = 32
num_heads = 2
ff_dim = 32

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

inputs_layer = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs_layer)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)

model = keras.Model(inputs=inputs_layer, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(inputs, targets, epochs=50, batch_size=2)

# Word generation function
def generate_text(seed_text, next_words=10):
    result = seed_text.split()
    for _ in range(next_words):
        input_seq = [word2idx.get(w, 0) for w in result[-maxlen:]]
        input_seq = np.pad(input_seq, (maxlen - len(input_seq), 0))
        pred = model.predict(input_seq.reshape(1, maxlen), verbose=0)
        next_word = idx2word[np.argmax(pred)]
        result.append(next_word)
    return ' '.join(result)

print(generate_text("the quick brown fox", next_words=10))
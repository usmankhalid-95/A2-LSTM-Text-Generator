import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import re

# Define LSTM Model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        embed = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))


# Load model and tokenizer
@st.cache_resource
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint["vocab"]

    # Use hardcoded parameters based on your saved configuration
    vocab_size = len(vocab)
    embed_dim = 256       # Fixed to your saved parameter
    hidden_dim = 256      # Fixed to your saved parameter
    num_layers = 1        # Fixed to your saved parameter
    dropout = 0.3         # Fixed to your saved parameter

    model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


# Tokenizer functions
def tokenizer(text):
    return re.findall(r'\b\w+\b|[.,!?;]', text.lower())


class Vocabulary:
    def __init__(self, tokens):
        self.counter = Counter(tokens)
        self.itos = ['<unk>', '<eos>'] + sorted(self.counter.keys(), key=lambda k: -self.counter[k])
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos


# Generate text
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = F.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return ' '.join(tokens)


# Streamlit App
st.title("LSTM Text Generator")  # Moved to the very top

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a model:",
    ("Harry Potter", "Pride and Prejudice")
)

# Model paths
model_paths = {
    "Harry Potter": "harry_potter_lstm.pt",
    "Pride and Prejudice": "pride_prejudice_lstm.pt"
}

device = torch.device("mps" if torch.has_mps else "cpu")  # Use MPS if available, otherwise fallback to CPU

# Load the selected model
st.sidebar.write(f"Loading {model_choice} model...")
model, vocab = load_model(model_paths[model_choice], device)
st.sidebar.write("Model loaded!")

prompt = st.text_input("Enter a prompt:", value="Once upon a time")
max_seq_len = st.slider("Maximum sequence length:", min_value=10, max_value=100, value=30)
temperature = st.slider("Temperature (controls randomness):", min_value=0.5, max_value=1.5, value=1.0)

# Generate text on button click
if st.button("Generate"):
    st.write("Generating text...")
    generated_text = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device)
    st.subheader("Generated Text")
    st.write(generated_text)


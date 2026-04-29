"""
Bidirectional LSTM with multi‑head self‑attention for ETF prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, embed_dim)
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        if return_attention:
            return out, attn.mean(dim=1)  # average over heads
        return out


class LSTMAttentionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128, num_layers=2,
                 bidirectional=True, dropout=0.2, num_heads=4):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = MultiHeadSelfAttention(lstm_out_dim, num_heads)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_out_dim, output_dim)

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)                     # (B, T, lstm_out_dim)
        if return_attention:
            attn_out, attn_weights = self.attention(lstm_out, return_attention=True)
        else:
            attn_out = self.attention(lstm_out)
        
        attn_out = self.layer_norm(lstm_out + attn_out)  # residual
        attn_out = self.dropout(attn_out)
        
        # Mean pooling over time
        pooled = attn_out.mean(dim=1)                   # (B, lstm_out_dim)
        out = self.head(pooled)                         # (B, output_dim)

        if return_attention:
            return out, attn_weights
        return out


class LSTMAttentionTrainer:
    def __init__(self, input_dim, output_dim, hidden_size=128, num_layers=2,
                 bidirectional=True, dropout=0.2, num_heads=4, lr=0.001, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMAttentionPredictor(
            input_dim, output_dim, hidden_size, num_layers,
            bidirectional, dropout, num_heads
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.output_dim = output_dim

    def fit(self, X, y, epochs=120, batch_size=64, patience=20):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(batch_X)

            avg_loss = train_loss / len(X)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X_latest):
        """Predict next-day returns and attention weights for the latest window."""
        self.model.eval()
        X_t = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, n_feat)
        with torch.no_grad():
            preds, attn_weights = self.model(X_t, return_attention=True)
        return preds.squeeze(0).cpu().numpy(), attn_weights.squeeze(0).cpu().numpy()  # (n_etfs,), (seq_len, seq_len)

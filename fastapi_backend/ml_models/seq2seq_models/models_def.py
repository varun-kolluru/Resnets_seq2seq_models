import torch
from torch import nn
import math



class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.xh = nn.Linear(embed_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

        nn.init.xavier_uniform_(self.xh.weight)
        nn.init.xavier_uniform_(self.hh.weight)

    def forward(self, src):
        """
        src: (batch, src_len)
        """
        batch_size, src_len = src.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=src.device)

        embedded = self.embedding(src)  # (batch, src_len, embed)

        for t in range(src_len):
            x_t = embedded[:, t, :]
            h_t = torch.tanh(self.xh(x_t) + self.hh(h_t))

        return h_t  # final hidden state

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.xh = nn.Linear(embed_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        nn.init.xavier_uniform_(self.xh.weight)
        nn.init.xavier_uniform_(self.hh.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, input_token, hidden):
        """
        input_token: (batch)
        hidden: (batch, hidden_size)
        """
        embedded = self.embedding(input_token)  # (batch, embed)

        hidden = torch.tanh(self.xh(embedded) + self.hh(hidden))
        output = self.fc_out(hidden)  # logits

        return output, hidden

class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        hidden = self.encoder(src)

        input_token = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs 



class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Gates: input, forget, output, candidate
        self.x2h = nn.Linear(embed_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.hidden_size = hidden_size

        nn.init.xavier_uniform_(self.x2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)

    def forward(self, src):
        """
        src: (batch, src_len)
        """
        batch_size, src_len = src.shape

        h_t = torch.zeros(batch_size, self.hidden_size, device=src.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=src.device)

        embedded = self.embedding(src)  # (batch, src_len, embed)

        for t in range(src_len):
            x_t = embedded[:, t, :]

            gates = self.x2h(x_t) + self.h2h(h_t)
            i, f, o, g = gates.chunk(4, dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)

        return h_t, c_t

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.x2h = nn.Linear(embed_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        nn.init.xavier_uniform_(self.x2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, input_token, hidden, cell):
        """
        input_token: (batch)
        hidden: (batch, hidden_size)
        cell: (batch, hidden_size)
        """
        embedded = self.embedding(input_token)

        gates = self.x2h(embedded) + self.h2h(hidden)
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        output = self.fc_out(hidden)

        return output, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        hidden, cell = self.encoder(src)

        input_token = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs
    

class AttnRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.xh = nn.Linear(embed_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.hidden_size = hidden_size

        nn.init.xavier_uniform_(self.xh.weight)
        nn.init.xavier_uniform_(self.hh.weight)

    def forward(self, src):
        """
        src: (batch, src_len)
        """
        batch_size, src_len = src.shape
        device = src.device

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs = []

        embedded = self.embedding(src)  # (batch, src_len, embed)

        for t in range(src_len):
            x_t = embedded[:, t, :]
            h_t = torch.tanh(self.xh(x_t) + self.hh(h_t))
            outputs.append(h_t.unsqueeze(1))

        # (batch, src_len, hidden)
        outputs = torch.cat(outputs, dim=1)

        return outputs, h_t

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, src_len, hidden)
        decoder_hidden: (batch, hidden)
        """
        src_len = encoder_outputs.size(1)

        dec_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(dec_hidden)
        )

        # (batch, src_len)
        attention = self.v(energy).squeeze(2)

        attn_weights = torch.softmax(attention, dim=1)

        # context: (batch, hidden)
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attn_weights

class AttnRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention

        self.xh = nn.Linear(embed_size + hidden_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        nn.init.xavier_uniform_(self.xh.weight)
        nn.init.xavier_uniform_(self.hh.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        input_token: (batch)
        hidden: (batch, hidden)
        encoder_outputs: (batch, src_len, hidden)
        """
        embedded = self.embedding(input_token)  # (batch, embed)

        context, attn_weights = self.attention(encoder_outputs, hidden)

        rnn_input = torch.cat((embedded, context), dim=1)

        hidden = torch.tanh(self.xh(rnn_input) + self.hh(hidden))

        output = self.fc_out(hidden)

        return output, hidden, attn_weights

class AttnRNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        encoder_outputs, hidden = self.encoder(src)

        input_token = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(
                input_token, hidden, encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs
    

class AttnLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.x2h = nn.Linear(embed_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.hidden_size = hidden_size

        nn.init.xavier_uniform_(self.x2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)

    def forward(self, src):
        """
        src: (batch, src_len)
        """
        batch_size, src_len = src.shape
        device = src.device

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)

        embedded = self.embedding(src)
        outputs = []

        for t in range(src_len):
            x_t = embedded[:, t, :]

            gates = self.x2h(x_t) + self.h2h(h_t)
            i, f, o, g = gates.chunk(4, dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, src_len, hidden)

        return outputs, h_t, c_t

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, src_len, hidden)
        decoder_hidden: (batch, hidden)
        """
        src_len = encoder_outputs.size(1)

        dec_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(dec_hidden)
        )

        attention = self.v(energy).squeeze(2)
        attn_weights = torch.softmax(attention, dim=1)

        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attn_weights

class AttnLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention

        self.x2h = nn.Linear(embed_size + hidden_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        nn.init.xavier_uniform_(self.x2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        """
        input_token: (batch)
        hidden: (batch, hidden)
        cell: (batch, hidden)
        encoder_outputs: (batch, src_len, hidden)
        """
        embedded = self.embedding(input_token)

        context, attn_weights = self.attention(encoder_outputs, hidden)

        lstm_input = torch.cat((embedded, context), dim=1)

        gates = self.x2h(lstm_input) + self.h2h(hidden)
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        output = self.fc_out(hidden)

        return output, hidden, cell, attn_weights

class AttnLSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input_token = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs



class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        emb_dim,
        n_heads,
        n_layers,
        ff_dim,
        dropout,
        pad_idx,
        device
    ):
        super().__init__()

        self.device = device
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, emb_dim)

        # 🔥 SEPARATE positional encoders
        self.src_pos_encoder = PositionalEncoding(emb_dim, dropout=dropout)
        self.trg_pos_encoder = PositionalEncoding(emb_dim, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)

    def make_src_key_padding_mask(self, src):
        return src == self.pad_idx

    def make_trg_key_padding_mask(self, trg):
        return trg == self.pad_idx

    def make_trg_subsequent_mask(self, trg_len):
        mask = torch.triu(
            torch.ones(trg_len, trg_len, device=self.device),
            diagonal=1
        )
        return mask.bool()  # True = masked

    def forward(self, src, trg):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """

        src_pad_mask = self.make_src_key_padding_mask(src)
        trg_pad_mask = self.make_trg_key_padding_mask(trg)
        trg_mask = self.make_trg_subsequent_mask(trg.size(1))

        src_emb = self.src_pos_encoder(self.src_embedding(src))
        trg_emb = self.trg_pos_encoder(self.trg_embedding(trg))

        out = self.transformer(
            src_emb,
            trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc_out(out)










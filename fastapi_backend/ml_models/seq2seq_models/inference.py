import torch
import spacy
import torchtext
from torchtext.data import Field

from ml_models.seq2seq_models.models_def import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= spaCy =================
spacy_de = spacy.load("de_core_news_sm")

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

# ================= Fields =================
SRC = Field(
    tokenize=tokenize_de,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    batch_first=True
)

TRG = Field(
    tokenize=lambda x: x.split(),
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    batch_first=True
)

# ================= Load vocab safely =================
with torch.serialization.safe_globals([torchtext.vocab.Vocab]):
    SRC.vocab = torch.load(
        "ml_models/seq2seq_models/vocab/src_vocab.pt",
        weights_only=False
    )
    TRG.vocab = torch.load(
        "ml_models/seq2seq_models/vocab/trg_vocab.pt",
        weights_only=False
    )

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 256

# ================= MODEL FACTORY =================
def build_model(model_name: str):

    if model_name == "rnn":
        enc = RNNEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
        dec = RNNDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)
        model = RNNSeq2Seq(enc, dec, device)

    elif model_name == "lstm":
        enc = LSTMEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
        dec = LSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)
        model = LSTMSeq2Seq(enc, dec, device)

    elif model_name == "attn_rnn":
        attn = BahdanauAttention(HID_DIM)
        enc = AttnRNNEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
        dec = AttnRNNDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attn)
        model = AttnRNNSeq2Seq(enc, dec, device)

    elif model_name == "attn_lstm":
        attn = BahdanauAttention(HID_DIM)
        enc = AttnLSTMEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
        dec = AttnLSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attn)
        model = AttnLSTMSeq2Seq(enc, dec, device)

    elif model_name == "transformer":
        model = TransformerSeq2Seq(
            src_vocab_size=INPUT_DIM,
            trg_vocab_size=OUTPUT_DIM,
            emb_dim=256,
            n_heads=4,
            n_layers=2,
            ff_dim=512,
            dropout=0.1,
            pad_idx=TRG.vocab.stoi["<pad>"],
            device=device
        )

    else:
        raise ValueError("Invalid model name")

    return model.to(device)

def load_model(model_name: str):
    model = build_model(model_name)

    checkpoint = torch.load(
        f"ml_models/seq2seq_models/trained_models/{model_name}.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def translate_sentence(sentence, model, model_name, max_len=50):

    model.eval()  # disable dropout / batchnorm

    tokens = tokenize_de(sentence)
    tokens = ["<sos>"] + tokens + ["<eos>"]

    src_idx = [
        SRC.vocab.stoi.get(tok, SRC.vocab.stoi["<unk>"])
        for tok in tokens
    ]

    src_tensor = torch.LongTensor([src_idx]).to(device)

    trg_indexes = [TRG.vocab.stoi["<sos>"]]

    with torch.no_grad():

        # ===== RNN =====
        if model_name == "rnn":
            hidden = model.encoder(src_tensor)

            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                output, hidden = model.decoder(trg_tensor, hidden)
                pred = output.argmax(1).item()
                trg_indexes.append(pred)
                if pred == TRG.vocab.stoi["<eos>"]:
                    break

        # ===== LSTM =====
        elif model_name == "lstm":
            hidden, cell = model.encoder(src_tensor)

            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
                pred = output.argmax(1).item()
                trg_indexes.append(pred)
                if pred == TRG.vocab.stoi["<eos>"]:
                    break

        # ===== Attention RNN =====
        elif model_name == "attn_rnn":
            encoder_outputs, hidden = model.encoder(src_tensor)

            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                output, hidden, _ = model.decoder(
                    trg_tensor, hidden, encoder_outputs
                )
                pred = output.argmax(1).item()
                trg_indexes.append(pred)
                if pred == TRG.vocab.stoi["<eos>"]:
                    break

        # ===== Attention LSTM =====
        elif model_name == "attn_lstm":
            encoder_outputs, hidden, cell = model.encoder(src_tensor)

            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                output, hidden, cell, _ = model.decoder(
                    trg_tensor, hidden, cell, encoder_outputs
                )
                pred = output.argmax(1).item()
                trg_indexes.append(pred)
                if pred == TRG.vocab.stoi["<eos>"]:
                    break

        # ===== Transformer (FIXED) =====
        elif model_name == "transformer":
            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes]).to(device)

                output = model(src_tensor, trg_tensor)

                pred = output[:, -1, :].argmax(dim=-1).item()
                trg_indexes.append(pred)

                if pred == TRG.vocab.stoi["<eos>"]:
                    break

    trg_tokens = [
        TRG.vocab.itos[i]
        for i in trg_indexes[1:-1]
    ]

    return " ".join(trg_tokens)



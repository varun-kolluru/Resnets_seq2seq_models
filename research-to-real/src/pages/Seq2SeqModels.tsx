import { useState } from "react";
import { ChevronDown, Languages } from "lucide-react";
import LiveInteractionDemo_seq2seq from "@/components/LiveInteractionDemo_seq2seq";

const models = [
  { id: "rnn", name: "RNN" },
  { id: "lstm", name: "LSTM" },
  { id: "attention_rnn", name: "RNN + Attention" },
  { id: "attention_lstm", name: "LSTM + Attention" },
  { id: "transformer", name: "Transformer" },
];

const Seq2SeqModels = () => {
  const [showVisuals, setShowVisuals] = useState(false);
  const [showDataset, setShowDataset] = useState(false);
  const [showHparams, setShowHparams] = useState(false);
  const [showDemo, setShowDemo] = useState(true);

  return (
    <div className="container mx-auto px-6 py-8">

      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-4 mb-3">
          <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20">
            <Languages className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold">
              Sequence-to-Sequence Models
            </h1>
            <p className="text-muted-foreground">
              German → English Neural Machine Translation
            </p>
          </div>
        </div>
      </div>

      {/* Research Papers */}
      <div className="glass-card rounded-xl px-6 py-4 mb-6">
        <h2 className="text-lg font-semibold mb-2">Research Papers Studied</h2>
        <ul className="grid md:grid-cols-2 gap-x-6 gap-y-1 text-sm text-muted-foreground">
          <li>Bahdanau et al., 2015 — Attention-based NMT</li>
          <li>Vaswani et al., 2017 — Transformer</li>
        </ul>
      </div>

      {/* Models Implemented */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Models Implemented</h2>
        <ul className="grid sm:grid-cols-2 md:grid-cols-3 gap-3 text-sm text-muted-foreground">
          {models.map(m => (
            <li key={m.id}>• {m.name}</li>
          ))}
        </ul>
      </div>

      {/* Loss & BLEU Visualizations */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <button
          onClick={() => setShowVisuals(!showVisuals)}
          className="flex items-center justify-between w-full"
        >
          <h2 className="text-xl font-semibold">
            Loss & BLEU Visualizations
          </h2>
          <ChevronDown
            className={`transition-transform ${showVisuals ? "rotate-180" : ""}`}
          />
        </button>

        {showVisuals && (
          <div className="mt-6 space-y-10 max-h-[700px] overflow-y-auto pr-2">
            {models.map(model => (
              <div key={model.id} className="space-y-4">
                <h3 className="text-lg font-semibold">
                  {model.name}
                </h3>

                <div className="grid md:grid-cols-2 gap-6">
                  <img
                    src={`/images/seq2seq/${model.id}_loss.png`}
                    alt={`${model.name} Loss`}
                    className="rounded-xl border border-border"
                  />

                  <img
                    src={`/images/seq2seq/${model.id}_belu.png`}
                    alt={`${model.name} BLEU`}
                    className="rounded-xl border border-border"
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Live Interaction Demo */}
      <div className="relative rounded-2xl p-6 mb-10
                      border border-primary/50
                      bg-gradient-to-br from-primary/20 via-primary/10 to-accent/20
                      shadow-lg shadow-primary/20">

        <button
          onClick={() => setShowDemo(!showDemo)}
          className="flex items-center justify-between w-full"
        >
          <h2 className="text-xl font-semibold text-primary flex items-center gap-2">
            <span className="text-green-500">✔</span>
            Live Translation Demo
          </h2>
          <ChevronDown className={`transition-transform ${showDemo ? "rotate-180" : ""}`} />
        </button>

        {showDemo && (
          <div className="mt-6 space-y-4 text-muted-foreground">
            <LiveInteractionDemo_seq2seq />
          </div>
        )}
      </div>

      {/* Dataset */}
      <div className="glass-card rounded-2xl p-6 mb-6">
        <button
          onClick={() => setShowDataset(!showDataset)}
          className="flex items-center justify-between w-full"
        >
          <h2 className="text-xl font-semibold">Dataset Used</h2>
          <ChevronDown className={`transition-transform ${showDataset ? "rotate-180" : ""}`} />
        </button>

        {showDataset && (
          <p className="mt-4 text-muted-foreground">
            German–English dataset from Kaggle with tokenization,
            vocabulary truncation, and padding.
          </p>
        )}
      </div>

      {/* Hyperparameters */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <button
          onClick={() => setShowHparams(!showHparams)}
          className="flex items-center justify-between w-full"
        >
          <h2 className="text-xl font-semibold">Training Configuration</h2>
          <ChevronDown className={`transition-transform ${showHparams ? "rotate-180" : ""}`} />
        </button>

        {showHparams && (
          <div className="mt-4 space-y-2 text-muted-foreground">
            <p>• Embedding Dim: <b>256</b></p>
            <p>• Hidden Size: <b>256</b></p>
            <p>• Optimizer: Adam</p>
            <p>• Loss: Cross-Entropy</p>
            <p>• Metric: BLEU</p>
          </div>
        )}
      </div>

    </div>
  );
};

export default Seq2SeqModels;

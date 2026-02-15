import { useState } from "react";
import { ArrowRight, Loader2 } from "lucide-react";

const MODELS = [
  { id: "rnn", label: "RNN" },
  { id: "lstm", label: "LSTM" },
  { id: "attn_rnn", label: "Attention RNN" },
  { id: "attn_lstm", label: "Attention LSTM" },
  { id: "transformer", label: "Transformer" },
];

const LiveInteractionDemo_seq2seq = () => {
  const [model, setModel] = useState("transformer");
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleTranslate = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setError("");
    setOutputText("");

    try {
      const res = await fetch("http://127.0.0.1:8000/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          text: inputText,
        }),
      });

      if (!res.ok) throw new Error("Translation failed");

      const data = await res.json();
      setOutputText(data.translation);
    } catch (err) {
      setError("Unable to translate. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-6 rounded-2xl border border-border bg-background/70 p-6 backdrop-blur-md">

      {/* Model Selector */}
      <div className="mb-4">
        <label className="block text-xs text-muted-foreground mb-1">
          Select Model
        </label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
        >
          {MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs text-muted-foreground mb-1">
          German Input
        </label>
        <textarea
          rows={3}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="z.B. Ich liebe maschinelles Lernen"
          className="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>

      {/* Action */}
      <button
        onClick={handleTranslate}
        disabled={loading}
        className="flex items-center justify-center gap-2 w-full rounded-xl bg-primary px-4 py-2 text-primary-foreground font-medium hover:opacity-90 transition disabled:opacity-60"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Translating...
          </>
        ) : (
          <>
            Translate
            <ArrowRight className="w-4 h-4" />
          </>
        )}
      </button>

      {/* Output */}
      {(outputText || error) && (
        <div className="mt-5 rounded-xl border border-border bg-muted/50 p-4">
          <p className="text-xs text-muted-foreground mb-1">
            English Translation
          </p>

          {error ? (
            <p className="text-sm text-red-500">{error}</p>
          ) : (
            <p className="text-sm text-foreground leading-relaxed">
              {outputText}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default LiveInteractionDemo_seq2seq;

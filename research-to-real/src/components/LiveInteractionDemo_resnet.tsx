import { useState } from "react";

const LiveInteractionDemo_resnet = () => {
  const [attackType, setAttackType] = useState("gaussian");
  const [noiseValue, setNoiseValue] = useState(0);
  const [modelName, setModelName] = useState("resnet34");
  const [loading, setLoading] = useState(false);

  const [inputImage, setInputImage] = useState<string | null>(null);
  const [noisyImage, setNoisyImage] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);

  const [predictedClass, setPredictedClass] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  /* ---------- Image Upload ---------- */
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setInputImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  /* ---------- Actual FastAPI Call ---------- */
    const runModel = async () => {
    if (!inputImage) return;

    setLoading(true);
    setPredictedClass(null);
    setConfidence(null);
    setNoisyImage(null);
    setOutputImage(null);

    try {
        // Clean the base64 string if it has data URL prefix
        let cleanImage = inputImage;
        if (inputImage.includes('base64,')) {
        cleanImage = inputImage.split('base64,')[1];
        }

        const res = await fetch("http://127.0.0.1:8000/infer", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model_name: modelName,
            attack_type: attackType,
            noise_value: noiseValue,
            image: cleanImage, // Send only base64 without data URL prefix
        }),
        });

        if (!res.ok) {
        const errorData = await res.json();
        console.error("Server error:", errorData);
        throw new Error(errorData.error || `HTTP ${res.status}`);
        }

        const data = await res.json();
        console.log(data);

        setPredictedClass(data.predicted_class);
        setConfidence(data.confidence);
        setNoisyImage(data.noisy_image);
        setOutputImage(data.output_image);

    } catch (err) {
        console.error("FastAPI error:", err);
        alert(`Error: ${err.message}`);
    } finally {
        setLoading(false);
    }
    };

  
  return (
    <>
      {/* ================= PANELS ================= */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* ================= INPUT IMAGE ================= */}
        <div className="rounded-xl border border-border p-4 bg-background/60">
          <h3 className="text-sm font-semibold mb-3">Input Image</h3>

          <div className="h-56 rounded-lg border border-border mb-4
                          flex items-center justify-center bg-muted/30 overflow-hidden">
            {inputImage ? (
              <img src={inputImage} className="h-full object-contain" />
            ) : (
              <span className="text-xs text-muted-foreground">
                No image uploaded
              </span>
            )}
          </div>

          <label className="block text-center">
            <span
              className="text-sm px-4 py-2 rounded-lg
                         bg-primary text-primary-foreground
                         font-medium cursor-pointer
                         shadow-md shadow-primary/30
                         hover:opacity-90 transition inline-block"
            >
              Upload Image
            </span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleImageUpload}
            />
          </label>
        </div>

        {/* ================= ATTACK CONFIG ================= */}
        <div className="rounded-xl border border-border p-4 bg-background/60">

        {/* Attack Type */}
        <label className="text-xs text-muted-foreground mb-1 block">
        Attack Type
        </label>
        <select
        value={attackType}
        onChange={(e) => {
            const type = e.target.value;
            setAttackType(type);
            setNoiseValue(0); // ✅ start from 0
        }}
        className="w-full mb-4 rounded-lg
                    border-2 border-primary/60
                    bg-primary/10 p-2 text-sm
                    focus:outline-none focus:ring-2 focus:ring-primary"
        >
        <option value="gaussian">Gaussian Noise</option>
        <option value="fgsm">FGSM Attack</option>
        <option value="weight">Weight Perturbation</option>
        </select>

        {/* Slider */}
        <label className="text-xs text-muted-foreground mb-1 block">
        Attack Strength: <b>{noiseValue}</b>
        </label>
        <input
        type="range"
        min={0}                                     // ✅ start from 0
        max={
            attackType === "gaussian" ? 0.1 :
            attackType === "fgsm" ? 0.3 : 0.05
        }
        step={0.01}
        value={noiseValue}
        onChange={(e) => setNoiseValue(Number(e.target.value))}
        className="w-full"
        />


          {/* Noisy Image */}
          <div className="mt-4 h-56 rounded-lg border border-border
                          bg-muted/30 flex items-center justify-center overflow-hidden">
            {noisyImage ? (
              <img src={noisyImage} className="h-full object-contain" />
            ) : (
              <span className="text-xs text-muted-foreground">
                Noisy image (from server)
              </span>
            )}
          </div>
        </div>

        {/* ================= OUTPUT ================= */}
        <div className="rounded-xl border border-border p-4 bg-background/60">
          <h3 className="text-sm font-semibold mb-3">Model Output</h3>

          <div className="h-56 rounded-lg border border-border
                          bg-muted/30 flex items-center justify-center overflow-hidden">
            {outputImage ? (
              <img src={outputImage} className="h-full object-contain" />
            ) : (
              <span className="text-xs text-muted-foreground">
                Output image
              </span>
            )}
          </div>

          <div className="mt-4 text-sm space-y-1">
            <p>
              <span className="text-muted-foreground">Predicted Class:</span>{" "}
              <b>{predictedClass ?? "—"}</b>
            </p>
            <p>
              <span className="text-muted-foreground">Confidence:</span>{" "}
              <b>
                {confidence !== null
                  ? `${(confidence * 100).toFixed(2)}%`
                  : "—"}
              </b>
            </p>
          </div>
        </div>
      </div>

      {/* ================= CONTROLS ================= */}
      <div className="mt-6 flex flex-wrap items-center gap-4">

        <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="rounded-md border border-border
                     bg-background p-2 text-sm"
        >
          <option value="resnet34">ResNet-34</option>
          <option value="adversarial_resnet">Adversarial ResNet</option>
          <option value="sam_resnet">SAM ResNet</option>
        </select>

        <button
          onClick={runModel}
          disabled={!inputImage || loading}
          className="px-6 py-2 rounded-lg bg-primary
                     text-primary-foreground font-medium
                     hover:opacity-90 transition disabled:opacity-50"
        >
          Run Model
        </button>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
            Running inference…
          </div>
        )}
      </div>
    </>
  );
};

export default LiveInteractionDemo_resnet;

import { useState } from 'react';
import { Eye,  ChevronDown} from 'lucide-react';
const modelOptions = [
  { id: 'resnet34', name: 'ResNet-34' },
  { id: 'adv_resnet34', name: 'Adversarial ResNet-34' },
  { id: 'sam_resnet34', name: 'SAM ResNet-34' },
];

const VisionModels = () => {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [showDataset, setShowDataset] = useState(false);
  const [showHparams, setShowHparams] = useState(false);

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-12">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20">
            <Eye className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground">Vision Models</h1>
            <p className="text-muted-foreground">Computer vision architectures for image understanding</p>
          </div>
        </div>
      </div>

     {/* Research Papers */}
      <div className="glass-card rounded-2xl p-6 mb-8 space-y-4">
        <h2 className="text-xl font-semibold">Research Papers Studied</h2>
        <ul className="list-disc list-inside text-muted-foreground space-y-1">
          <li>Deep Residual Learning for Image Recognition ~ (2015, Kaiming He)</li>
          <li>Intriguing Properties of Neural Networks ~ (2014, Christian Szegedy)</li>
          <li>Explaining and Harnessing Adversarial Examples ~ (2015, Ian Goodfellow)</li>
          <li>
            Sharpness-Aware Minimization for Efficiently Improving Generalization ~
            (2021, Pierre Foret)
          </li>
        </ul>
      </div>

      {/* Dataset Accordion */}
      <div className="glass-card rounded-2xl p-6 mb-6">
        <button
          onClick={() => setShowDataset(!showDataset)}
          className="flex items-center justify-between w-full text-left"
        >
          <h2 className="text-xl font-semibold">Dataset Used</h2>
          <ChevronDown className={`transition-transform ${showDataset ? 'rotate-180' : ''}`} />
        </button>

        {showDataset && (
          <p className="mt-4 text-muted-foreground leading-relaxed">
            CIFAR-10 dataset from <code>torchvision.datasets</code> containing
            50,000 training and 10,000 test images of shape (32×32×3).
            <br /><br />
            This dataset was chosen due to limited computational resources
            (single NVIDIA P100 GPU). Training multiple ResNet variants,
            including adversarial and SAM-based models, on larger datasets
            would be computationally prohibitive for comparative analysis.
          </p>
        )}
      </div>

      {/* Hyperparameters Accordion */}
      <div className="glass-card rounded-2xl p-6 mb-10">
        <button
          onClick={() => setShowHparams(!showHparams)}
          className="flex items-center justify-between w-full text-left"
        >
          <h2 className="text-xl font-semibold">Hyperparameters Used</h2>
          <ChevronDown className={`transition-transform ${showHparams ? 'rotate-180' : ''}`} />
        </button>

        {showHparams && (
          <div className="mt-4 text-muted-foreground space-y-2">
            <p>• Batch Size: <b>256</b></p>
            <p>
              • Optimizer: SGD
              <br />
              <code>
                optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
              </code>
            </p>
            <p>
              • Epochs: <b>30</b>
              <br />
              <span className="text-sm">
                (40 epochs for adversarial ResNet since loss does not saturate)
              </span>
            </p>
          </div>
        )}
      </div>

      {/* Model Dropdown */}
      <div className="glass-card rounded-2xl p-6 mb-10 max-w-md">
        <label className="block text-xs text-muted-foreground uppercase tracking-wider mb-2">
          Choose Model
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full rounded-lg bg-secondary/60 border border-border p-3 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="">Select a ResNet variant</option>
          {modelOptions.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </div>

      {/* Placeholder for future content */}
      <div className="glass-card rounded-2xl p-12 text-center">
        {selectedModel ? (
          <p className="text-muted-foreground text-lg">
            You selected <b>{modelOptions.find(m => m.id === selectedModel)?.name}</b>.
            <br />
            Model-specific analysis and visualizations will appear here.
          </p>
        ) : (
          <>
            <Eye className="w-16 h-16 text-muted-foreground/30 mx-auto mb-4" />
            <p className="text-muted-foreground text-lg">
              Select a model above to begin exploration
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default VisionModels;

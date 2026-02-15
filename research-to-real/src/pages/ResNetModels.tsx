import { useState } from 'react';
import { Layers, ChevronDown } from 'lucide-react';
import LiveInteractionDemo from "@/components/LiveInteractionDemo_resnet";
import LiveInteractionDemo_resnet from '@/components/LiveInteractionDemo_resnet';


const ResNetModels = () => {
  const [showDataset, setShowDataset] = useState(false);
  const [showHparams, setShowHparams] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [showDemo, setShowDemo] = useState(true);


  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-4 mb-3">
          <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20">
            <Layers className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground">
              ResNet Models
            </h1>
            <p className="text-muted-foreground">
              CNNs with skip connections
            </p>
          </div>
        </div>
      </div>

      {/* Research Papers – flatter */}
      <div className="glass-card rounded-xl px-6 py-4 mb-6">
        <h2 className="text-lg font-semibold mb-2">Research Papers Studied</h2>
        <ul className="grid md:grid-cols-2 gap-x-6 gap-y-1 text-sm text-muted-foreground">
          <li>Deep Residual Learning for Image Recognition (2015, K. He)</li>
          <li>Intriguing Properties of Neural Networks (2014, C. Szegedy)</li>
          <li>Explaining & Harnessing Adversarial Examples (2015, I. Goodfellow)</li>
          <li>Sharpness-Aware Minimization (2021, P. Foret)</li>
        </ul>
      </div>

      {/* Comparative Analysis */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="flex items-center justify-between w-full"
        >
          <h2 className="text-xl font-semibold">Comparative Analysis</h2>
          <ChevronDown className={`transition-transform ${showComparison ? 'rotate-180' : ''}`} />
        </button>

        {showComparison && (
          <div className="mt-6 space-y-8 max-h-[600px] overflow-y-auto pr-2">
            {/* Results Image */}
            <div>
              <h3 className="font-semibold mb-3">Results</h3>
              <img
                src="/images/resnet/Resnet_analysis.png"
                alt="ResNet Results"
                className="rounded-xl border border-border"
              />
            </div>

            {/* Loss Landscapes */}
            <div>
              <h3 className="font-semibold mb-3">3D Loss Landscape Visualization</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <img src="/images/resnet/loss_resnet.png" className="rounded-xl border border-border" />
                <img src="/images/resnet/loss_adv.png" className="rounded-xl border border-border" />
                <img src="/images/resnet/loss_sam.png" className="rounded-xl border border-border" />
              </div>
            </div>

            {/* Observations*/}
            <div>
              <h3 className="font-semibold mb-3">Observations</h3>
              <ul className="list-disc pl-6 space-y-2 text-sm text-muted-foreground">
                <li>
                  <strong>SAM-ResNet achieves the highest clean accuracy</strong>, indicating stronger
                  generalization to unseen data.
                </li>
                <li>
                  <strong>Under high Gaussian noise and FGSM attacks, Adversarial ResNet performs better</strong>,
                  which aligns with its robustness-focused training.
                </li>
                <li>
                  <strong>With weight noise perturbations, SAM-ResNet still maintains higher accuracy</strong>,
                  although the relative degradation is similar across all models.
                </li>
                <li>
                  <strong>Loss landscape analysis highlights structural differences</strong>: Adversarial
                  ResNet shows a rougher surface, standard ResNet remains smoother, while SAM-ResNet
                  exhibits a flatter minimum—directly demonstrating SAM’s objective.
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* Live Interaction – highlighted & open by default */}
      <div className="relative rounded-2xl p-6 mb-10 
                      border border-primary/50 
                      bg-gradient-to-br from-primary/20 via-primary/10 to-accent/20
                      shadow-lg shadow-primary/20
                      animate-pulse-subtle">

        {/* Glow overlay */}
        <div className="absolute inset-0 rounded-2xl 
                        bg-primary/10 blur-2xl -z-10" />

        <button
          onClick={() => setShowDemo(!showDemo)}
          className="flex items-center justify-between w-full"
        >
          <h2 className="text-xl font-semibold text-primary flex items-center gap-2">
            <span className="text-green-500 text-xl">✔</span>
            Live Interaction
          </h2>

          <ChevronDown
            className={`transition-transform ${showDemo ? 'rotate-180' : ''}`}
          />
        </button>

        {showDemo && (
          <div className="mt-6 text-muted-foreground space-y-3">
            <p className="font-medium text-foreground">
              Interactively explore ResNet robustness
            </p>
            <ul className="list-disc list-inside space-y-1">
              <li>Upload an image and visualize predictions</li>
              <li>Apply noise, FGSM & weight perturbation attacks</li>
              <li>Observe confidence shifts across models</li>
            </ul>

            <LiveInteractionDemo_resnet />
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
          <ChevronDown className={`transition-transform ${showDataset ? 'rotate-180' : ''}`} />
        </button>

        {showDataset && (
          <p className="mt-4 text-muted-foreground leading-relaxed">
            CIFAR-10 dataset from <code>torchvision.datasets</code> with
            50k train and 10k test images of size (32×32×3).
            <br /><br />
            Chosen due to limited compute (single NVIDIA P100 GPU) while enabling
            fair comparative analysis across robustness techniques.
          </p>
        )}
      </div>

      {/* Hyperparameters */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <button
          onClick={() => setShowHparams(!showHparams)}
          className="flex items-center justify-between w-full"
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
              <code>lr=0.1, momentum=0.9, weight_decay=5e-4</code>
            </p>
            <p>
              • Epochs: <b>30</b> (40 for adversarial ResNet)
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResNetModels;

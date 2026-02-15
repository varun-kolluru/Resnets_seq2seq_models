import { Link } from 'react-router-dom';
import { Layers, Cpu, Eye, ArrowRight } from 'lucide-react';

const categories = [
  {
    path: '/resnet-models',
    title: 'ResNet Models',
    description: 'Residual Networks for testing Robustness and Generalization capabilities of CNNs',
    icon: Layers,
    count: 3,
  },
  {
    path: '/seq2seq-models',
    title: 'Seq2Seq Models',
    description: 'Sequence-to-sequence architectures for BLEU comparision in translation of Text from German to English',
    icon: Cpu,
    count: 5,
  },
  {
    path: '/vision-models',
    title: 'Vision Models',
    description: 'Use of CTC Loss and Vision Transformers for Hand Writing Text Recognition',
    icon: Eye,
    count: 2,
  },
];

const ModelsHome = () => {
  return (
    <div className="container mx-auto px-6 py-12">
      <div className="text-center mb-16">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="gradient-text">Explore Models</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Select a category to explore implemented architectures, hyperparameters, and comparative analysis.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {categories.map((category, index) => {
          const Icon = category.icon;
          return (
            <Link
              key={category.path}
              to={category.path}
              className="glass-card rounded-2xl p-8 hover:border-primary/50 transition-all duration-300 group hover:glow-effect opacity-0 animate-fade-in"
              style={{ animationDelay: `${0.2 + index * 0.15}s` }}
            >
              <div className="p-4 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 w-fit mb-6 group-hover:from-primary/30 group-hover:to-accent/30 transition-all duration-300">
                <Icon className="w-8 h-8 text-primary" />
              </div>
              
              <h2 className="text-2xl font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                {category.title}
              </h2>
              
              <p className="text-muted-foreground mb-6 leading-relaxed">
                {category.description}
              </p>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground font-mono">
                  {category.count} models
                </span>
                <div className="flex items-center gap-2 text-primary group-hover:gap-3 transition-all">
                  <span className="text-sm font-medium">Explore</span>
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
};

export default ModelsHome;

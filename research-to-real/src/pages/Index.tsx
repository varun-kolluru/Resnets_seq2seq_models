import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowRight, Brain, Sparkles, Code, Zap } from 'lucide-react';
import GradientBackground from '@/components/GradientBackground';

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: 'Deep Learning Research',
      description: 'Studied cutting-edge research papers on neural network architectures',
    },
    {
      icon: Code,
      title: 'From Scratch pytorch Implementations',
      description: 'Every model built from the ground up for complete understanding',
    },
    {
      icon: Zap,
      title: 'End-to-End ML Pipeline',
      description: 'Fully deployed models with interactive testing capabilities',
    },
    {
      icon: Sparkles,
      title: 'Comparative Analysis',
      description: 'Side-by-side architecture comparisons and performance benchmarks',
    },
  ];

  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      <GradientBackground />

      <div className="relative z-10 container mx-auto px-6 min-h-screen flex flex-col items-center justify-center">
        {/* Hero Section */}
        <div className="text-center max-w-4xl mx-auto">
          <div 
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-8 opacity-0 animate-fade-in"
            style={{ animationDelay: '0.2s' }}
          >
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">Deep Learning Research Platform</span>
          </div>

          <h1 
            className="text-5xl md:text-7xl font-bold mb-6 opacity-0 animate-fade-in"
            style={{ animationDelay: '0.4s' }}
          >
            <span className="gradient-text">Research</span>
            <span className="text-foreground">2</span>
            <span className="gradient-text">Models</span>
          </h1>

          <p 
            className="text-xl md:text-2xl text-muted-foreground mb-4 opacity-0 animate-fade-in max-w-3xl mx-auto"
            style={{ animationDelay: '0.6s' }}
          >
            From research papers to production-ready implementations.
          </p>

          <p 
            className="text-lg text-muted-foreground/80 mb-12 opacity-0 animate-fade-in max-w-2xl mx-auto leading-relaxed"
            style={{ animationDelay: '0.8s' }}
          >
            Explore deep learning architectures implemented from scratch, tested on real datasets, 
            and deployed as interactive end-to-end ML projects with comprehensive comparative analysis.
          </p>

          <div 
            className="opacity-0 animate-fade-in"
            style={{ animationDelay: '1s' }}
          >
            <Button
              variant="hero"
              size="xl"
              onClick={() => navigate('/models')}
              className="group"
            >
              <span>Go</span>
              <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
            </Button>
          </div>
        </div>

        {/* Features Grid */}
        <div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-24 w-full max-w-6xl opacity-0 animate-fade-in"
          style={{ animationDelay: '1.2s' }}
        >
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={feature.title}
                className="glass-card rounded-2xl p-6 hover:border-primary/30 transition-all duration-300 group"
                style={{ animationDelay: `${1.4 + index * 0.1}s` }}
              >
                <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 w-fit mb-4 group-hover:from-primary/30 group-hover:to-accent/30 transition-all duration-300">
                  <Icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Index;

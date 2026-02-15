import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Brain, Layers, Eye, Cpu } from 'lucide-react';

const navItems = [
  { path: '/resnet-models', label: 'ResNet Models', icon: Layers },
  { path: '/seq2seq-models', label: 'Seq2Seq Models', icon: Cpu },
  { path: '/vision-models', label: 'Vision Models', icon: Eye },
];

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-border/50">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <Link 
            to="/models" 
            className="flex items-center gap-3 group"
          >
            <div className="p-2 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 group-hover:from-primary/30 group-hover:to-accent/30 transition-all duration-300">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            <span className="font-semibold text-lg gradient-text">Research2Models</span>
          </Link>

          <div className="flex items-center gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link key={item.path} to={item.path}>
                  <Button
                    variant={isActive ? 'navActive' : 'nav'}
                    size="sm"
                    className="gap-2"
                  >
                    <Icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </Button>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

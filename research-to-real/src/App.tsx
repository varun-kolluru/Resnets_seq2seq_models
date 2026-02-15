import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import ModelsHome from "./pages/ModelsHome";
import ResNetModels from "./pages/ResNetModels";
import Seq2SeqModels from "./pages/Seq2SeqModels";
import VisionModels from "./pages/VisionModels";
import ModelsLayout from "./layouts/ModelsLayout";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route element={<ModelsLayout />}>
            <Route path="/models" element={<ModelsHome />} />
            <Route path="/resnet-models" element={<ResNetModels />} />
            <Route path="/seq2seq-models" element={<Seq2SeqModels />} />
            <Route path="/vision-models" element={<VisionModels />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

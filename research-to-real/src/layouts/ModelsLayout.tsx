import { Outlet } from 'react-router-dom';
import Navbar from '@/components/Navbar';

const ModelsLayout = () => {
  return (
    <div className="min-h-screen bg-background neural-grid">
      <Navbar />
      <main className="pt-20 pb-12">
        <Outlet />
      </main>
    </div>
  );
};

export default ModelsLayout;

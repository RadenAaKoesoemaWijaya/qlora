import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Brain, 
  Database, 
  Settings, 
  Activity, 
  Save, 
  BarChart3,
  Menu,
  X,
  Cpu
} from 'lucide-react';

const Layout = () => {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const navigation = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Models', href: '/models', icon: Brain },
    { name: 'Model Management', href: '/models/manage', icon: Cpu },
    { name: 'Datasets', href: '/datasets', icon: Database },
    { name: 'Configure Training', href: '/training/configure', icon: Settings },
    { name: 'Training Monitor', href: '/training/monitor', icon: Activity },
    { name: 'Checkpoints', href: '/checkpoints', icon: Save },
    { name: 'Evaluation', href: '/evaluation', icon: BarChart3 },
  ];

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? 'w-64' : 'w-0'
        } bg-indigo-950 transition-all duration-300 overflow-hidden flex-shrink-0`}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="p-6 border-b border-indigo-900">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-sky-500 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-white font-bold text-lg">TechnoFriendR</h1>
                <p className="text-indigo-300 text-xs">AI Fine-tuning Platform</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.href || 
                (item.href !== '/' && location.pathname.startsWith(item.href));
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  data-testid={`nav-${item.name.toLowerCase().replace(/\s+/g, '-')}`}
                  className={`${
                    isActive
                      ? 'bg-indigo-900 text-white'
                      : 'text-indigo-300 hover:bg-indigo-900/50 hover:text-white'
                  } group flex items-center px-3 py-2.5 text-sm font-medium rounded-md transition-colors`}
                >
                  <Icon className="mr-3 h-5 w-5 flex-shrink-0" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-indigo-900">
            <div className="bg-indigo-900 rounded-lg p-3">
              <p className="text-xs text-indigo-300 mb-1">AI Fine-tuning Platform</p>
              <p className="text-xs text-indigo-400"> 2026 PT. ASMER SAHABAT SUKSES</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
          <button
            data-testid="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-md hover:bg-slate-100 transition-colors"
          >
            {sidebarOpen ? (
              <X className="h-5 w-5 text-slate-600" />
            ) : (
              <Menu className="h-5 w-5 text-slate-600" />
            )}
          </button>

          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm font-medium text-slate-900">TechnoFriendR Team</p>
              <p className="text-xs text-slate-500">AI Fine-tuning Solutions</p>
            </div>
            <div className="w-10 h-10 bg-sky-500 rounded-full flex items-center justify-center">
              <span className="text-white font-semibold text-sm">TF</span>
            </div>
          </div>
        </div>

        {/* Page Content */}
        <div className="flex-1 overflow-auto">
          <div className="p-6 md:p-8">
            <Outlet />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Layout;
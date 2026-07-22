import React, { useEffect } from 'react';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';

export default function Toast({ toast, onClose }) {
  if (!toast || !toast.message) return null;

  const { message, type = 'error' } = toast;

  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 4000);
    return () => clearTimeout(timer);
  }, [toast, onClose]);

  const configs = {
    error: {
      bg: 'bg-white border-red-200 text-red-700',
      icon: AlertCircle,
      iconColor: 'text-red-500',
    },
    success: {
      bg: 'bg-white border-emerald-200 text-emerald-700',
      icon: CheckCircle2,
      iconColor: 'text-emerald-500',
    },
    info: {
      bg: 'bg-white border-brand-200 text-brand-700',
      icon: Info,
      iconColor: 'text-brand-500',
    }
  };

  const config = configs[type] || configs.error;
  const Icon = config.icon;

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-bounce-in max-w-sm w-full">
      <div className={`p-4 rounded-2xl border ${config.bg} bg-white flex items-center justify-between space-x-3 transition-all duration-300`}>
        <div className="flex items-center space-x-3">
          <Icon className={`w-5 h-5 ${config.iconColor} shrink-0`} />
          <p className="text-sm font-medium leading-snug">{message}</p>
        </div>

        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 p-1 rounded-lg transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

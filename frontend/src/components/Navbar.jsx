import React from 'react';
import { FileText, HelpCircle } from 'lucide-react';

export default function Navbar({ langChoice, setLangChoice, activePage, setActivePage }) {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-[81%] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between relative">

        {/* Left: Logo Text Only */}
        <div className="flex items-center -ml-6 cursor-pointer" onClick={() => setActivePage('summarize')}>
          <span className="text-2xl font-extrabold text-brand-600 tracking-tight">
            Summarizer.
          </span>
        </div>

        {/* Center: Navigation Pill Tabs (Positioned in exact center of header) */}
        <nav className="absolute left-1/2 -translate-x-1/2 flex items-center space-x-1 bg-gray-100/80 p-1 rounded-full border border-gray-200">
          <button
            onClick={() => setActivePage('summarize')}
            className={`flex items-center space-x-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              activePage === 'summarize'
                ? 'bg-white text-brand-600 shadow-sm border border-gray-200/80'
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Tóm tắt</span>
          </button>

          <button
            onClick={() => setActivePage('guide')}
            className={`flex items-center space-x-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              activePage === 'guide'
                ? 'bg-white text-brand-600 shadow-sm border border-gray-200/80'
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Hướng dẫn sử dụng</span>
          </button>
        </nav>

        {/* Right: Language Selector */}
        <div className="flex items-center space-x-2 bg-gray-50 border border-gray-200 rounded-full px-3.5 py-1.5 text-sm text-gray-600">
          <span className="font-semibold text-gray-500 text-sm">Ngôn ngữ:</span>
          <select
            value={langChoice}
            onChange={(e) => setLangChoice(e.target.value)}
            className="bg-transparent outline-none cursor-pointer text-sm font-semibold text-gray-800 font-['Inter',sans-serif]"
          >
            <option value="auto" className="bg-white text-gray-800 font-medium py-1 font-['Inter',sans-serif]">Tự động</option>
            <option value="vi" className="bg-white text-gray-800 font-medium py-1 font-['Inter',sans-serif]">Tiếng Việt</option>
            <option value="en" className="bg-white text-gray-800 font-medium py-1 font-['Inter',sans-serif]">English</option>
          </select>
        </div>

      </div>
    </header>
  );
}

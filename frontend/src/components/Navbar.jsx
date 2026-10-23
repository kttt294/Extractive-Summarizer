import React from 'react';
import { FileText, HelpCircle } from 'lucide-react';

export default function Navbar({ langChoice, setLangChoice, activePage, setActivePage }) {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="w-full lg:max-w-[86%] mx-auto px-3 sm:px-6 lg:px-8 py-2 lg:py-0 min-h-14 lg:h-16 grid grid-cols-[auto_minmax(0,1fr)_auto] sm:grid-cols-[1fr_auto_1fr] items-center gap-x-1.5 sm:gap-x-3">

        {/* Left: Logo & Title */}
        <button type="button" className="min-w-0 flex items-center justify-self-start gap-1.5 cursor-pointer" onClick={() => setActivePage('summarize')} aria-label="Về trang tóm tắt">
          <img
            src="/logo.png"
            alt="Logo"
            className="h-7 sm:h-8 w-auto shrink-0 object-contain transition-transform hover:scale-105"
          />
          <span className="hidden min-[480px]:inline text-base lg:text-xl font-extrabold text-brand-600 tracking-tight whitespace-nowrap">
            Summarizer.
          </span>
        </button>

        {/* Compact single-row navigation without absolute positioning. */}
        <nav aria-label="Điều hướng chính" className="min-w-0 justify-self-center flex items-center gap-0.5 sm:gap-1 bg-gray-100/80 p-0.5 sm:p-1 rounded-full border border-gray-200">
          <button
            onClick={() => setActivePage('summarize')}
            aria-current={activePage === 'summarize' ? 'page' : undefined}
            className={`min-h-7 sm:min-h-10 lg:min-h-0 flex items-center justify-center gap-1.5 px-1 sm:px-3 py-1 sm:py-2 lg:py-1.5 rounded-full text-[11px] sm:text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
              activePage === 'summarize'
                ? 'bg-white text-brand-600 shadow-sm border border-gray-200/80'
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <FileText className="hidden lg:block w-3.5 h-3.5 shrink-0" />
            <span>Tóm tắt</span>
          </button>

          <button
            onClick={() => setActivePage('guide')}
            aria-current={activePage === 'guide' ? 'page' : undefined}
            className={`min-h-7 sm:min-h-10 lg:min-h-0 flex items-center justify-center gap-1.5 px-1 sm:px-3 py-1 sm:py-2 lg:py-1.5 rounded-full text-[11px] sm:text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
              activePage === 'guide'
                ? 'bg-white text-brand-600 shadow-sm border border-gray-200/80'
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <HelpCircle className="hidden lg:block w-3.5 h-3.5 shrink-0" />
            <span>Hướng dẫn<span className="hidden lg:inline"> sử dụng</span></span>
          </button>
        </nav>

        {/* Right: Language Selector */}
        <div className="col-start-3 row-start-1 justify-self-end min-w-0 flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-full px-1.5 sm:px-2.5 py-1 sm:py-1.5 text-sm text-gray-600">
          <span className="hidden xl:inline font-semibold text-gray-500 text-sm">Ngôn ngữ:</span>
          <select
            value={langChoice}
            onChange={(e) => setLangChoice(e.target.value)}
            aria-label="Ngôn ngữ"
            className="min-w-0 max-w-full min-h-5 sm:min-h-7 bg-transparent outline-none cursor-pointer text-xs sm:text-sm font-semibold text-gray-800 font-['Inter',sans-serif]"
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

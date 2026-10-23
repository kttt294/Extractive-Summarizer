import React from 'react';
import { FileText, Link, Search } from 'lucide-react';

export default function InputTabs({ activeTab, setActiveTab, text, setText, url, setUrl, topic, setTopic }) {
  const tabs = [
    { id: 'text', label: 'Đoạn văn', icon: FileText },
    { id: 'url', label: 'Link URL', icon: Link },
    { id: 'topic', label: 'Chủ đề', icon: Search }
  ];

  return (
    <div className="space-y-4">
      {/* Pill-shaped Filter Tabs */}
      <div className="grid grid-cols-3 gap-1.5 sm:flex sm:flex-wrap sm:gap-2">
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`min-w-0 min-h-11 sm:min-h-0 flex items-center justify-center gap-1 sm:gap-1.5 px-1.5 sm:px-4 py-2 rounded-full text-xs sm:text-sm whitespace-nowrap font-medium border transition-all ${
                isActive
                  ? 'bg-brand-500 text-white border-brand-500 shadow-sm'
                  : 'bg-white text-gray-500 border-gray-200 hover:border-gray-300 hover:text-gray-700'
              }`}
            >
              <Icon className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* Fixed Height Input Area */}
      <div className={`flex flex-col justify-start ${activeTab === 'text' ? 'h-52 sm:h-[260px]' : 'lg:h-[260px]'}`}>
        {activeTab === 'text' && (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Dán đoạn văn bản hoặc bài viết cần tóm tắt vào đây..."
            rows={10}
            className="w-full h-full bg-white border border-gray-200 rounded-xl p-4 text-base sm:text-sm text-gray-700 placeholder-gray-400 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all resize-none"
          />
        )}

        {activeTab === 'url' && (
          <div className="space-y-2.5">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Nhập link bài báo..."
              className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3.5 text-base sm:text-sm text-gray-700 placeholder-gray-400 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all"
            />
          </div>
        )}

        {activeTab === 'topic' && (
          <div className="space-y-2.5">
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Nhập chủ đề..."
              className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3.5 text-base sm:text-sm text-gray-700 placeholder-gray-400 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all"
            />
          </div>
        )}
      </div>
    </div>
  );
}

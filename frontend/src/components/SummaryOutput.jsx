import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

export default function SummaryOutput({ sentences, highlightIndices, activeHoverIndex, setActiveHoverIndex }) {
  const [copied, setCopied] = useState(false);

  if (!sentences || sentences.length === 0) return null;

  const handleCopy = () => {
    const textToCopy = sentences.map((s, i) => `${i + 1}. ${s}`).join('\n');
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-card p-5 space-y-4">
      {/* Card Header */}
      <div className="flex items-center justify-between border-b border-gray-100 pb-3">
        <div className="flex items-center space-x-2">
          <h2 className="text-sm font-bold text-gray-800">Bản Tóm Tắt Trích Xuất</h2>
        </div>

        <button
          onClick={handleCopy}
          title={copied ? "Đã sao chép!" : "Sao chép"}
          className="flex items-center justify-center p-2 text-gray-400 hover:text-brand-500 bg-gray-50 hover:bg-brand-50 rounded-full border border-gray-200 hover:border-brand-200 transition-all cursor-pointer"
        >
          {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
        </button>
      </div>

      {/* Extracted Sentences List */}
      <div className="space-y-2">
        {sentences.map((sent, idx) => {
          const originalIdx = highlightIndices ? highlightIndices[idx] : null;
          const isHovered = activeHoverIndex === originalIdx;

          return (
            <div
              key={idx}
              onMouseEnter={() => setActiveHoverIndex(originalIdx)}
              onMouseLeave={() => setActiveHoverIndex(null)}
              className={`p-3 rounded-xl border transition-all cursor-pointer ${
                isHovered
                  ? 'bg-brand-50 border-brand-300 text-brand-700 shadow-sm'
                  : 'bg-gray-50 border-gray-100 text-gray-600 hover:border-gray-200 hover:bg-white'
              }`}
            >
              <div className="flex items-start space-x-3">
                <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold shrink-0 mt-0.5 ${
                  isHovered ? 'bg-brand-500 text-white' : 'bg-brand-100 text-brand-600'
                }`}>
                  {idx + 1}
                </span>
                <p className="text-sm leading-relaxed">{sent}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

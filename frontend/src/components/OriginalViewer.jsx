import React from 'react';

export default function OriginalViewer({ title, originalText, highlightIndices, activeHoverIndex, setActiveHoverIndex }) {
  if (!originalText) return null;

  const paragraphs = originalText.split('\n').filter(p => p.trim());

  let globalSentenceCounter = 0;

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-card p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-gray-100 pb-3 gap-3">
        <h2 className="text-sm font-bold text-gray-800 flex-1 min-w-0 leading-snug">
          {title ? title : 'Văn Bản Gốc'}
        </h2>
        <span className="text-[11px] text-gray-400 font-medium shrink-0 pt-0.5 whitespace-nowrap">
          Di chuột để xem câu tương ứng
        </span>
      </div>

      {/* Article Content */}
      <div className="max-h-[420px] overflow-y-auto pr-2 space-y-3 text-sm text-gray-600 leading-relaxed">
        {paragraphs.map((para, pIdx) => {
          const sentences = para.match(/[^.!?]+[.!?]+/g) || [para];

          return (
            <p key={pIdx} className="space-x-0.5">
              {sentences.map((sent, sIdx) => {
                const currentSentIndex = globalSentenceCounter++;
                const isSelected = highlightIndices && highlightIndices.includes(currentSentIndex);
                const isHovered = activeHoverIndex === currentSentIndex;

                return (
                  <span
                    key={sIdx}
                    onMouseEnter={() => setActiveHoverIndex(currentSentIndex)}
                    onMouseLeave={() => setActiveHoverIndex(null)}
                    className={`transition-all duration-200 cursor-pointer rounded px-0.5 py-0.5 ${
                      isHovered
                        ? 'bg-brand-500 text-white font-semibold'
                        : isSelected
                        ? 'bg-brand-50 text-brand-700 font-medium border-b-2 border-brand-300'
                        : 'hover:bg-gray-100'
                    }`}
                  >
                    {sent}{' '}
                  </span>
                );
              })}
            </p>
          );
        })}
      </div>
    </div>
  );
}

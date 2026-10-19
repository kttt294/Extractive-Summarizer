import React from 'react';

export default function OriginalViewer({ title, originalText, sentenceSpans = [], highlightIndices = [], activeHoverIndex, setActiveHoverIndex }) {
  if (!originalText) return null;
  const pieces = [];
  let cursor = 0;
  for (const span of sentenceSpans) {
    if (span.start < cursor || span.end > originalText.length || span.end <= span.start) continue;
    pieces.push(originalText.slice(cursor, span.start));
    const selected = highlightIndices.includes(span.index);
    const hovered = activeHoverIndex === span.index;
    pieces.push(
      <span key={span.index}
        onMouseEnter={() => setActiveHoverIndex(span.index)}
        onMouseLeave={() => setActiveHoverIndex(null)}
        className={`transition-all duration-200 cursor-pointer rounded px-0.5 py-0.5 ${
          hovered ? 'bg-brand-500 text-white font-semibold'
          : selected ? 'bg-brand-50 text-brand-700 font-medium border-b-2 border-brand-300'
          : 'hover:bg-gray-100'}`}>
        {originalText.slice(span.start, span.end)}
      </span>
    );
    cursor = span.end;
  }
  pieces.push(originalText.slice(cursor));
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-card p-5 space-y-4">
      <div className="border-b border-gray-100 pb-3">
        <h2 className="text-sm font-bold text-gray-800 leading-snug">{title || 'Văn Bản Gốc'}</h2>
      </div>
      <div className="max-h-[420px] overflow-y-auto pr-2 text-sm text-gray-600 leading-relaxed whitespace-pre-wrap">
        {pieces}
      </div>
    </div>
  );
}

import React from 'react';

export default function StatsHeader({ stats }) {
  if (!stats) return null;

  const items = [
    {
      label: 'Tỷ lệ nén',
      value: `-${stats.compression_ratio}%`,
      sub: `${stats.summary_sentence_count} / ${stats.original_sentence_count} câu`,
      accent: 'text-brand-600',
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Thời gian xử lý',
      value: `${stats.latency_ms} ms`,
      sub: `Ngôn ngữ: ${stats.detected_language}`,
      accent: 'text-emerald-600',
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Silhouette Score',
      value: stats.silhouette_score,
      sub: 'Độ phân tách cụm',
      accent: 'text-violet-600',
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Diversity Score',
      value: stats.diversity_score,
      sub: 'Độ đa dạng ngữ nghĩa',
      accent: 'text-amber-600',
      bg: 'bg-white border-gray-200'
    }
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {items.map((it, idx) => {
        return (
          <div key={idx} className={`p-3.5 rounded-xl border ${it.bg}`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500">{it.label}</span>
            </div>
            <div className={`text-xl font-bold mt-1 ${it.accent}`}>{it.value}</div>
            <div className="text-[11px] text-gray-400 mt-0.5">{it.sub}</div>
          </div>
        );
      })}
    </div>
  );
}

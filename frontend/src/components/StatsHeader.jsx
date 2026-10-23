import React from 'react';

function formatNumber(value, maximumFractionDigits = 4) {
  if (!Number.isFinite(value)) return '—';
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits,
    useGrouping: false,
  }).format(value);
}

export default function StatsHeader({ stats }) {
  if (!stats) return null;

  const items = [
    {
      label: 'Tỷ lệ nén',
      value: `-${formatNumber(stats.compression_ratio, 1)}%`,
      sub: `${stats.summary_sentence_count} / ${stats.original_sentence_count} câu`,
      accent: 'text-brand-600',
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Thời gian xử lý',
      value: `${formatNumber(stats.latency_ms, 1)} ms`,
      sub: `Ngôn ngữ: ${stats.detected_language}`,
      accent: 'text-emerald-600',
      bg: 'bg-white border-gray-200'
    },
    {
      label: stats.silhouette_score == null ? 'Phương pháp' : 'Silhouette Score',
      value: stats.silhouette_score == null ? 'Logistic Regression' : formatNumber(stats.silhouette_score),
      sub: stats.silhouette_score == null ? 'Biểu diễn câu bằng SBERT' : 'Độ phân tách cụm',
      accent: 'text-violet-600',
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Diversity Score',
      value: formatNumber(stats.diversity_score),
      sub: stats.diversity_score == null ? 'Cần ít nhất 2 câu' : 'Độ đa dạng ngữ nghĩa',
      accent: 'text-amber-600',
      bg: 'bg-white border-gray-200'
    }
  ];

  return (
    <div className="grid min-w-0 grid-cols-2 xl:grid-cols-4 gap-3">
      {items.map((it, idx) => {
        return (
          <div key={idx} className={`min-w-0 [overflow-wrap:anywhere] p-3.5 rounded-xl border ${it.bg}`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500">{it.label}</span>
            </div>
            <div className={`text-xl font-bold tabular-nums mt-1 ${it.accent}`}>{it.value}</div>
            <div className="text-[11px] text-gray-400 mt-0.5">{it.sub}</div>
          </div>
        );
      })}
    </div>
  );
}

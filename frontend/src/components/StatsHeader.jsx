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
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Thời gian xử lý',
      value: `${formatNumber(stats.latency_ms, 1)} ms`,
      bg: 'bg-white border-gray-200'
    },
    {
      label: stats.silhouette_score == null ? 'Phương pháp' : 'Silhouette Score',
      value: stats.silhouette_score == null ? 'Logistic Regression' : formatNumber(stats.silhouette_score),
      sub: stats.silhouette_score == null ? null : 'Độ phân tách cụm',
      valueSize: stats.silhouette_score == null ? 'text-sm leading-7' : 'text-xl',
      bg: 'bg-white border-gray-200'
    },
    {
      label: 'Diversity Score',
      value: formatNumber(stats.diversity_score),
      sub: stats.diversity_score == null ? 'Cần ít nhất 2 câu' : null,
      bg: 'bg-white border-gray-200'
    }
  ];

  return (
    <div className="grid min-w-0 grid-cols-2 xl:grid-cols-4 gap-3">
      {items.map((it, idx) => {
        return (
          <div key={idx} className={`min-w-0 [overflow-wrap:anywhere] p-3.5 rounded-xl border text-gray-700 text-center ${it.bg}`}>
            <div className="flex items-center justify-center">
              <span className="text-xs font-medium">{it.label}</span>
            </div>
            <div className={`${it.valueSize ?? 'text-xl'} font-bold tabular-nums mt-1`}>{it.value}</div>
            {it.sub && <div className="text-[11px] mt-0.5">{it.sub}</div>}
          </div>
        );
      })}
    </div>
  );
}

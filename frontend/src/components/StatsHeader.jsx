import React from 'react';
import { Gauge, Clock, Layers, Sparkles, Calendar, Info } from 'lucide-react';

export default function StatsHeader({ stats, notice, dateRange, isTodayNews }) {
  if (!stats) return null;

  const items = [
    {
      label: 'Tỷ lệ nén',
      value: `-${stats.compression_ratio}%`,
      sub: `${stats.summary_sentence_count} / ${stats.original_sentence_count} câu`,
      icon: Gauge,
      accent: 'text-brand-600',
      bg: 'bg-brand-50 border-brand-100'
    },
    {
      label: 'Thời gian xử lý',
      value: `${stats.latency_ms} ms`,
      sub: `Ngôn ngữ: ${stats.detected_language}`,
      icon: Clock,
      accent: 'text-emerald-600',
      bg: 'bg-emerald-50 border-emerald-100'
    },
    {
      label: 'Silhouette Score',
      value: stats.silhouette_score,
      sub: 'Độ phân tách cụm',
      icon: Layers,
      accent: 'text-violet-600',
      bg: 'bg-violet-50 border-violet-100'
    },
    {
      label: 'Diversity Score',
      value: stats.diversity_score,
      sub: 'Độ đa dạng ngữ nghĩa',
      icon: Sparkles,
      accent: 'text-amber-600',
      bg: 'bg-amber-50 border-amber-100'
    }
  ];

  return (
    <div className="space-y-3">
      {/* Timeline Badge & Older Articles Fallback Warning Banner */}
      {(dateRange || notice) && (
        <div className="bg-white rounded-xl border border-gray-200 p-3.5 space-y-2 text-xs">
          {dateRange && (
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-gray-700">
                <Calendar className="w-4 h-4 text-brand-500 shrink-0" />
                <span className="font-semibold">Khoảng thời gian bài viết được tổng hợp:</span>
              </div>
              <span className="font-bold text-brand-600 bg-brand-50 border border-brand-100 px-2.5 py-1 rounded-full">
                {dateRange}
              </span>
            </div>
          )}

          {notice && (
            <div className="flex items-start space-x-2 text-amber-700 bg-amber-50 border border-amber-200 p-2.5 rounded-lg pt-2">
              <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <span>{notice}</span>
            </div>
          )}
        </div>
      )}

      {/* Intrinsic Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {items.map((it, idx) => {
          const Icon = it.icon;
          return (
            <div key={idx} className={`p-3.5 rounded-xl border ${it.bg}`}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-500">{it.label}</span>
                <Icon className={`w-4 h-4 ${it.accent}`} />
              </div>
              <div className={`text-xl font-bold mt-1 ${it.accent}`}>{it.value}</div>
              <div className="text-[11px] text-gray-400 mt-0.5">{it.sub}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

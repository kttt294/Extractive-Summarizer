import React from 'react';
import { Calendar } from 'lucide-react';

export default function TopicBanner({ dateRange }) {
  if (!dateRange) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-3.5 text-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 [overflow-wrap:anywhere]">
        <div className="flex items-center space-x-2 text-gray-700">
          <Calendar className="w-4 h-4 text-brand-500 shrink-0" />
          <span className="font-semibold">Khoảng thời gian tổng hợp:</span>
        </div>
        <span className="max-w-full self-start sm:self-auto font-bold text-brand-600 bg-brand-50 border border-brand-100 px-2.5 py-1 rounded-full">
          {dateRange}
        </span>
      </div>
    </div>
  );
}

import React from 'react';
import { Calendar, Info } from 'lucide-react';

export default function TopicBanner({ dateRange, notice }) {
  if (!dateRange && !notice) return null;

  return (
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
  );
}

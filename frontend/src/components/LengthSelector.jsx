import React from 'react';

export default function LengthSelector({ length, setLength, langChoice }) {
  if (langChoice !== 'en') {
    return (
      <div className="space-y-2 mt-4 rounded-xl border border-brand-100 bg-brand-50 p-3">
        <p className="text-sm font-semibold text-brand-700">Tóm tắt tiếng Việt: 3 câu</p>
        <p className="text-xs text-gray-600 leading-relaxed">
          Chọn ba câu quan trọng và giữ thứ tự trong bài gốc. Văn bản ngắn sẽ giữ tối đa số câu hiện có.
          {langChoice === 'auto' && ' Với tiếng Anh, hệ thống dùng độ dài tiêu chuẩn.'}
        </p>
      </div>
    );
  }
  const options = [
    { id: 'brief', label: 'Ngắn gọn' },
    { id: 'medium', label: 'Tiêu chuẩn' },
    { id: 'detailed', label: 'Chi tiết' }
  ];

  return (
    <div className="space-y-2 mt-4">
      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Độ dài bản tóm tắt
      </label>
      <div className="flex gap-2">
        {options.map((opt) => {
          const isSelected = length === opt.id;
          return (
            <button
              key={opt.id}
              type="button"
              onClick={() => setLength(opt.id)}
              className={`flex-1 py-3 px-3 rounded-xl border text-center transition-all cursor-pointer ${
                isSelected
                  ? 'border-brand-500 bg-brand-50 text-brand-600 font-semibold shadow-sm'
                  : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300 hover:text-gray-700'
              }`}
            >
              <div className="text-sm font-medium">{opt.label}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

import React from 'react';
import { CheckCircle } from 'lucide-react';

export default function UserGuide({ onStartSummarizing }) {
  const steps = [
    {
      step: '01',
      title: 'Chọn Phương Thức Nhập Liệu',
      desc: 'Hệ thống hỗ trợ 3 cách nhập liệu linh hoạt:',
      items: [
        { label: 'Đoạn văn thô:', text: 'Dán trực tiếp văn bản bài báo bất kỳ (tối thiểu 50 ký tự).' },
        { label: 'Link URL:', text: 'Nhập đường link bài báo điện tử (VnExpress, Dân Trí, Tuổi Trẻ, CNN...).' },
        { label: 'Theo Chủ đề:', text: 'Nhập từ khóa chủ đề để tổng hợp từ nhiều bài viết.' }
      ]
    },
    {
      step: '02',
      title: 'Độ Dài Và Phương Pháp Tóm Tắt',
      desc: 'Phương pháp được hiển thị cùng mỗi kết quả:',
      items: [
        { label: 'Tiếng Việt:', text: 'SBERT + Logistic Regression: Ngắn gọn tối đa 2 câu, Tiêu chuẩn 3 câu, Chi tiết 5 câu; bài ngắn giữ tối đa số câu hợp lệ hiện có.' },
        { label: 'Tiếng Anh:', text: 'Giữ phương pháp SBERT + K-Means kế thừa, có ba mức độ dài.' }
      ]
    },
    {
      step: '03',
      title: 'Phân Tích Chỉ Số Nội Tại (Intrinsic Metrics)',
      desc: 'Các chỉ số mô tả bản tóm tắt, không thay thế đánh giá với bản tham chiếu:',
      items: [
        { label: 'Silhouette Score:', text: 'Chỉ hiển thị với phương pháp K-Means của tiếng Anh; không áp dụng cho Logistic.' },
        { label: 'Diversity Score:', text: 'Mô tả sự khác biệt giữa embedding các câu được chọn; không bảo đảm loại hết ý trùng hoặc giữ đủ thông tin.' }
      ]
    },
    {
      step: '04',
      title: 'Tính Năng Interactive Highlighting',
      desc: 'Trải nghiệm tương tác trực quan:',
      items: [
        { label: 'Di chuột:', text: 'Rê chuột vào câu bất kỳ trong Bản Tóm Tắt → Câu tương ứng trong Văn Bản Gốc sẽ tự động phát sáng.' }
      ]
    }
  ];

  return (
    <div className="w-full lg:max-w-[86%] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 sm:space-y-8 animate-fade-in">
      {/* Title Header */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <h2 className="text-xl sm:text-2xl font-extrabold text-gray-900">Cách Sử Dụng Extractive Summarizer</h2>
      </div>

      {/* Grid of Steps */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((st, idx) => {
          return (
            <div key={idx} className="min-w-0 [overflow-wrap:anywhere] bg-white rounded-2xl border border-gray-200 shadow-card p-4 sm:p-6 space-y-4 hover:border-brand-200 transition-all">
              {/* Main Step Header (No Icon) */}
              <div className="flex items-start justify-between gap-3">
                <h3 className="text-base font-bold text-gray-800">{st.title}</h3>
                <span className="shrink-0 text-xl font-extrabold text-gray-300">{st.step}</span>
              </div>

              <p className="text-xs text-gray-500">{st.desc}</p>

              {/* Items List */}
              <ul className="space-y-2 pt-2 border-t border-gray-100">
                {st.items.map((it, iIdx) => (
                  <li key={iIdx} className="flex items-start space-x-2 text-xs text-gray-600 leading-relaxed">
                    <CheckCircle className="w-4 h-4 text-brand-500 shrink-0 mt-0.5" />
                    <span>
                      <strong className="text-gray-800 font-semibold">{it.label} </strong>
                      {it.text}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}

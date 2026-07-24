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
      title: 'Tùy Chỉnh Độ Dài Bản Tóm Tắt',
      desc: 'Lựa chọn mức tỷ lệ nén mong muốn:',
      items: [
        { label: 'Ngắn gọn:', text: 'Giữ lại các ý chính cốt lõi nhất, đọc nhanh trong 5 giây.' },
        { label: 'Tiêu chuẩn:', text: 'Tỷ lệ nén tối ưu nhất thu được từ thực nghiệm Grid Search.' },
        { label: 'Chi tiết:', text: 'Bao quát sâu các chi tiết, thích hợp cho bài báo nghiên cứu.' }
      ]
    },
    {
      step: '03',
      title: 'Phân Tích Chỉ Số Nội Tại (Intrinsic Metrics)',
      desc: 'Hệ thống tự động tính toán 2 chỉ số chất lượng:',
      items: [
        { label: 'Silhouette Score:', text: 'Đo độ phân tách ngữ nghĩa giữa các cụm K-Means Clustering.' },
        { label: 'Diversity Score:', text: 'Đo độ đa dạng thông tin, đảm bảo bản tóm tắt không bị lặp ý.' }
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
    <div className="max-w-[86%] mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 animate-fade-in">
      {/* Title Header */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <h2 className="text-2xl font-extrabold text-gray-900">Cách Sử Dụng Extractive Summarizer</h2>
      </div>

      {/* Grid of Steps */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((st, idx) => {
          return (
            <div key={idx} className="bg-white rounded-2xl border border-gray-200 shadow-card p-6 space-y-4 hover:border-brand-200 transition-all">
              {/* Main Step Header (No Icon) */}
              <div className="flex items-center justify-between">
                <h3 className="text-base font-bold text-gray-800">{st.title}</h3>
                <span className="text-xl font-extrabold text-gray-300">{st.step}</span>
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

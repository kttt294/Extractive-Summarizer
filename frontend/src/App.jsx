import React, { useState } from 'react';
import Navbar from './components/Navbar';
import InputTabs from './components/InputTabs';
import LengthSelector from './components/LengthSelector';
import TopicBanner from './components/TopicBanner';
import StatsHeader from './components/StatsHeader';
import SummaryOutput from './components/SummaryOutput';
import OriginalViewer from './components/OriginalViewer';
import UserGuide from './components/UserGuide';
import Toast from './components/Toast';
import { summarizeTextApi, summarizeUrlApi, summarizeTopicApi } from './api/client';
import { Loader2 } from 'lucide-react';

export default function App() {
  const [activePage, setActivePage] = useState('summarize'); // 'summarize' | 'guide'
  const [langChoice, setLangChoice] = useState('auto');
  const [activeTab, setActiveTab] = useState('text');
  const [length, setLength] = useState('medium');

  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [topic, setTopic] = useState('');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [toast, setToast] = useState(null);

  const [activeHoverIndex, setActiveHoverIndex] = useState(null);

  const handleSummarize = async () => {
    setToast(null);
    setLoading(true);
    setResult(null);

    try {
      let data = null;
      if (activeTab === 'text') {
        if (!text || text.trim().length < 50) {
          throw new Error('Vui lòng nhập đoạn văn bản dài tối thiểu 50 ký tự.');
        }
        data = await summarizeTextApi(text, langChoice, length);
      } else if (activeTab === 'url') {
        if (!url || !url.startsWith('http')) {
          throw new Error('Vui lòng nhập đường link bài báo hợp lệ (bắt đầu bằng http:// hoặc https://).');
        }
        data = await summarizeUrlApi(url, langChoice, length);
      } else if (activeTab === 'topic') {
        if (!topic || topic.trim().length < 2) {
          throw new Error('Vui lòng nhập từ khóa chủ đề cần tóm tắt.');
        }
        data = await summarizeTopicApi(topic, langChoice === 'auto' ? 'vi' : langChoice, length);
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || err.message || 'Đã có lỗi xảy ra trong quá trình tóm tắt.';
      setToast({ message: errorMsg, type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 flex flex-col relative">
      <Toast toast={toast} onClose={() => setToast(null)} />

      <Navbar
        langChoice={langChoice}
        setLangChoice={setLangChoice}
        activePage={activePage}
        setActivePage={setActivePage}
      />

      {/* Main Page Rendering */}
      {activePage === 'guide' ? (
        <UserGuide onStartSummarizing={() => setActivePage('summarize')} />
      ) : (
        <main className="flex-1 max-w-[86%] w-full mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-10 space-y-6">

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">

            {/* Left Column: Input Controller */}
            <div className="lg:col-span-5 bg-white rounded-2xl border border-gray-200 shadow-card p-6 h-[540px] flex flex-col justify-between">
              <div className="space-y-5">
                <InputTabs
                  activeTab={activeTab}
                  setActiveTab={setActiveTab}
                  text={text}
                  setText={setText}
                  url={url}
                  setUrl={setUrl}
                  topic={topic}
                  setTopic={setTopic}
                />

                <LengthSelector length={length} setLength={setLength} />
              </div>

              {/* Submit Button */}
              <button
                onClick={handleSummarize}
                disabled={loading}
                className="w-full py-3.5 px-4 bg-brand-500 hover:bg-brand-600 active:bg-brand-700 text-white font-semibold rounded-xl text-sm shadow-sm flex items-center justify-center space-x-2 transition-all disabled:opacity-50 cursor-pointer"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Đang xử lý...</span>
                  </>
                ) : (
                  <span>Tóm tắt ngay</span>
                )}
              </button>
            </div>

            {/* Right Column: Output Dashboard */}
            <div className="lg:col-span-7">
              {result ? (
                <div className="h-[540px] overflow-y-auto space-y-5 pr-1.5 scrollbar-thin scrollbar-thumb-slate-300">
                  {/* Topic Timeline & Notice Banner at VERY TOP */}
                  <TopicBanner dateRange={result.date_range} notice={result.notice} />

                  <SummaryOutput
                    sentences={result.summary_sentences}
                    highlightIndices={result.highlight_indices}
                    activeHoverIndex={activeHoverIndex}
                    setActiveHoverIndex={setActiveHoverIndex}
                  />

                  <OriginalViewer
                    title={result.title}
                    originalText={result.original_text}
                    highlightIndices={result.highlight_indices}
                    activeHoverIndex={activeHoverIndex}
                    setActiveHoverIndex={setActiveHoverIndex}
                  />

                  {/* 4 Stat Cards at VERY BOTTOM */}
                  <StatsHeader stats={result.stats} />
                </div>
              ) : (
                <div className="bg-white rounded-2xl border border-gray-200 shadow-card p-12 text-center space-y-2 h-[540px] flex flex-col items-center justify-center">
                  <div className="max-w-xs space-y-1.5">
                    <h3 className="text-base font-bold text-gray-800">Sẵn sàng tóm tắt</h3>
                    <p className="text-sm text-gray-400 leading-relaxed">
                      Nhập nội dung hoặc link bài báo ở cột bên trái, sau đó bấm <strong className="text-brand-500">Tóm tắt ngay</strong> để bắt đầu.
                    </p>
                  </div>
                </div>
              )}
            </div>

          </div>

        </main>
      )}
    </div>
  );
}

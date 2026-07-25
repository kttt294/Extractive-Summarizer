def write(doc, add_heading_1, add_heading_2, add_heading_3, add_p, add_bullet, add_code, style_table):
    add_heading_1("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT")

    add_heading_2("2.1. Tổng quan về Bài toán Tóm tắt văn bản (Text Summarization)")
    add_p("Tóm tắt văn bản tự động (Automatic Text Summarization) là một quá trình ứng dụng các thuật toán xử lý ngôn ngữ tự nhiên để rút gọn văn bản gốc thành một phiên bản ngắn gọn hơn, nhưng bắt buộc phải bảo tồn các thông tin quan trọng nhất, các sự kiện cốt lõi và tính logic của ngữ nghĩa gốc.")

    add_heading_2("2.2. Phân loại Bài toán Tóm tắt: Trích xuất (Extractive) và Sinh (Abstractive)")
    add_bullet("Extractive Summarization (Tóm tắt Trích xuất): Thuật toán sẽ quét qua toàn bộ các câu, tính toán điểm số tầm quan trọng (Importance Score), và \"trích xuất\" nguyên văn các câu đạt điểm cao nhất để ghép lại. Ưu điểm: Đảm bảo 100% tính chính xác sự thật (Factuality), không bịa đặt (Zero Hallucination).")
    add_bullet("Abstractive Summarization (Tóm tắt Sinh nội dung): Mô hình đọc hiểu toàn bộ văn bản và sinh ra các câu mới hoàn toàn từ đầu. Nhược điểm chí mạng là chi phí tính toán cực kỳ tốn kém và nguy cơ rất cao mắc phải hiện tượng Hallucination (tự bịa sự kiện, con số).")

    add_heading_2("2.3. Kiến trúc Đột phá Transformer")
    add_p("Năm 2017, bài báo lịch sử \"Attention Is All You Need\" (Vaswani et al.) đánh dấu sự ra đời của Transformer. Trước đó, các mô hình RNN (như LSTM, GRU) xử lý tuần tự (từng từ một) nên rất chậm và khó nhớ ngữ cảnh xa. Transformer khắc phục triệt để bằng cách thay thế hoàn toàn mạng hồi quy bằng cơ chế Tự chú ý (Self-Attention), cho phép khả năng song song hóa toàn diện.")
    add_p("Kiến trúc Transformer gồm 2 phần chính:")
    add_bullet("Encoder (Bộ mã hóa): Nhận câu đầu vào và chuyển thành vector ngữ nghĩa. Mục tiêu là đọc hiểu văn bản.")
    add_bullet("Decoder (Bộ giải mã): Nhận vector ngữ nghĩa và sinh ra từng từ một để tạo thành câu đầu ra. Mục tiêu là sinh văn bản.")
    
    add_heading_3("2.3.1. Cơ chế Self-Attention và Multi-Head Attention")
    add_p("Self-Attention cho phép mỗi từ trong câu đánh giá mức độ tương quan của nó với tất cả các từ khác cùng lúc. Toán học của Self-Attention xoay quanh 3 ma trận Q (Query), K (Key), V (Value):")
    add_code("Attention(Q, K, V) = softmax( (Q . K^T) / sqrt(d_k) ) . V")
    add_p("Thay vì dùng 1 hàm Attention, Transformer sử dụng Multi-head Attention chia vector ra thành H \"đầu\" (Heads) độc lập (thường là 8 head). Mỗi head học một đặc trưng ngữ nghĩa riêng biệt, sau đó nối lại (Concat) và nhân với ma trận trọng số cuối.")

    add_heading_3("2.3.2. Cấu trúc Luồng dữ liệu bên trong Encoder")
    add_p("Trước khi văn bản thực sự bước vào các khối Encoder, mỗi token đầu vào sẽ được ánh xạ thành một vector 768 chiều (đối với bản base). Vector khởi thủy này thực chất là phép cộng ma trận (element-wise addition) của 3 thành phần: (1) Token Embedding (mang ý nghĩa từ vựng), (2) Positional Embedding (đánh dấu vị trí thứ tự của từ trong câu), và (3) Segment Embedding (đánh dấu từ thuộc câu A hay câu B).")
    add_p("Mỗi lớp Encoder tiếp tục bao gồm các bước xử lý tinh vi: (1) Multi-head Self-attention để lấy ngữ cảnh. (2) Residual connection + LayerNorm để chống mất mát thông tin ban đầu. (3) Feed-forward network (ReLU) để trải phẳng và nén dữ liệu. (4) Tiếp tục Residual + LayerNorm. Mô hình tiêu biểu sử dụng toàn Encoder là BERT, RoBERTa.")

    add_heading_2("2.4. Mô hình Ngôn ngữ Hai chiều BERT (2018)")
    add_p("Năm 2018, BERT (Bidirectional Encoder Representations from Transformers) ra đời bằng cách xếp chồng các lớp Encoder của Transformer. Khác biệt cốt lõi là BERT học hai chiều (bidirectional) thông qua 2 tác vụ tiền huấn luyện: Masked Language Model (MLM - che đi một số từ và đoán lại) và Next Sentence Prediction (NSP - dự đoán 2 câu có liên tiếp không).")
    
    add_heading_2("2.5. Vấn đề Dị hướng (Anisotropy) của BERT")
    add_p("Khi áp dụng BERT vào bài toán so sánh độ tương đồng 2 câu, nếu ghép 2 câu lại thành chuỗi [CLS] Câu A [SEP] Câu B [SEP] (Cross-Encoder), BERT cho độ chính xác cực cao nhưng cực kỳ chậm. Nếu đem N=10,000 câu đi so sánh với nhau, BERT phải chạy qua mạng Neural 49.995.000 lần (mất 65 giờ).")
    add_p("Để nhanh hơn, người ta thử rút trích độc lập từng vector của câu (lấy tại token [CLS] hoặc tính MEAN tất cả các token) rồi tính Cosine Similarity. Tuy nhiên, kết quả lại tệ hại hơn cả mô hình GloVe cổ điển (theo điểm Spearman). Nguyên nhân là do Không gian vector của BERT bị mắc hội chứng Dị hướng (Anisotropy). Các vector không phân bố đều mà bị dồn ép vào một góc hẹp (hình nón dương). Hậu quả là tính Cosine giữa 2 câu ngẫu nhiên bất kỳ luôn ra điểm rất cao (0.8 - 0.9), khiến máy tính không thể phân biệt được câu giống và câu khác nhau.")

    add_heading_2("2.6. Giải pháp Đột phá: Sentence-BERT (Bi-Encoder)")
    add_p("Năm 2019, SBERT (Reimers & Gurevych) ra đời giải quyết bài toán tốc độ và hội chứng dị hướng. SBERT sử dụng kiến trúc mạng Siamese (mạng anh em sinh đôi), mã hóa độc lập 2 câu thành 2 vector cố định. Giúp giảm thời gian tìm kiếm từ 65 giờ xuống còn 5 giây.")
    
    add_heading_3("2.6.1. Lớp Toán gộp (Pooling Layer) và Xử lý Độ dài Câu")
    add_p("Về mặt kiến trúc, SBERT thừa hưởng giới hạn đầu vào của BERT là tối đa 512 tokens. Các câu ngắn hơn sẽ được chèn thêm token đệm ([PAD]), các câu dài hơn 512 tokens sẽ bị cắt cụt (truncate).")
    add_p("Đầu ra của BERT là một chuỗi các vector tương ứng với số lượng từ. Để giải quyết bài toán: 'Làm sao 2 câu có độ dài khác nhau lại có thể so sánh được với nhau?', SBERT tích hợp thêm Lớp Pooling ngay sau đầu ra của BERT để tính toán và cô đặc N vector (mỗi vector 768 chiều) thành chính xác 1 vector duy nhất (768 chiều). Nhóm tác giả thử nghiệm 3 chiến lược:")
    add_bullet("CLS-token: Chỉ lấy vector tại token [CLS]. Tuy nhiên, do token này vốn bị 'thiên kiến' (bias) bởi tác vụ Next Sentence Prediction trong quá trình pre-train của BERT, nó không đại diện tốt cho toàn bộ ý nghĩa của câu.")
    add_bullet("MAX Pooling: Lấy giá trị lớn nhất theo từng chiều không gian. Dù phát hiện từ khóa (keywords) tốt, chiến lược này làm vỡ cấu trúc ngữ cảnh của câu.")
    add_bullet("MEAN Pooling: Tính trung bình cộng tất cả vector token dựa trên chiều dài thực tế L của từng câu. Sử dụng Attention Mask để phớt lờ hoàn toàn các token đệm [PAD], nhờ đó biến L thành một tham số biến thiên (tùy thuộc vào số từ thật của mỗi câu) thay vì chia mù quáng cho max_length. Khi tính trung bình, các nhiễu tự triệt tiêu, để lại phần ngữ nghĩa cốt lõi nhất. Đây là chiến lược mặc định mang lại hiệu năng cao nhất trên mọi tập dữ liệu và cũng là cấu hình được sử dụng để tinh chỉnh mô hình Tiếng Việt trong Đồ án này.")

    add_heading_3("2.6.2. Các Hàm mục tiêu Huấn luyện (Objective Functions)")
    add_p("Quá trình huấn luyện đặc biệt này đã \"ép\" không gian vector của mô hình phải bung tỏa đều ra mọi hướng, nắn lại hình học không gian. Tùy vào loại dữ liệu, SBERT sử dụng 3 hàm mục tiêu:")
    add_bullet("a. Hàm mục tiêu Phân loại (Classification Objective): Sử dụng khi dữ liệu có nhãn hạng mục (ví dụ: mâu thuẫn, đồng thuận). SBERT trích 2 vector u, v, sau đó thực hiện phép nối (Concatenation): [u, v, |u - v|]. Thành phần |u - v| cực kỳ quan trọng vì nó trực tiếp đo lường khoảng cách hình học.")
    add_bullet("b. Hàm mục tiêu Hồi quy (Regression Objective): Dùng cho dữ liệu chấm điểm liên tục (ví dụ: điểm tương đồng 0-5). SBERT tính trực tiếp Cosine(u,v) và dùng hàm Mean-Squared-Error (MSE) để tối thiểu sai số.")
    add_bullet("c. Hàm mục tiêu Triplet (Triplet Objective): Nhận đầu vào 3 câu: Câu neo (Anchor a), Câu dương tính đồng nghĩa (Positive p), Câu âm tính khác nghĩa (Negative n). Hàm Triplet Loss tinh chỉnh mạng lưới sao cho khoảng cách Euclid ||a - p|| phải luôn nhỏ hơn ||a - n|| cộng thêm một khoảng biên an toàn (epsilon). Hàm này thể hiện rõ nhất cơ chế 'kéo gần, đẩy xa' trong học sâu.")

    add_heading_2("2.7. Đánh đổi giữa Tốc độ và Độ chính xác (Cross-Encoder vs Bi-Encoder)")
    add_p("Trong thiết kế hệ thống AI thực tế, luôn có sự đánh đổi (Trade-off) giữa Tốc độ và Độ chính xác:")
    add_bullet("Cross-Encoder (BERT): Ghép 2 câu lại với nhau ([CLS] Câu A [SEP] Câu B [SEP]). Nhờ cơ chế cross-attention, từ của 2 câu tương tác chéo với nhau ngay từ lớp mạng đầu tiên -> Độ chính xác hoàn hảo nhưng cực kỳ CHẬM.")
    add_bullet("Bi-Encoder (SBERT): Đưa từng câu vào mô hình một cách độc lập để trích xuất Embedding. Hai câu bị cô lập và chỉ tương tác ở phép tính khoảng cách Euclid hoặc Cosine cuối cùng -> Mất mát một phần thông tin nhưng cực kỳ NHANH.")
    
    add_heading_3("2.7.1. Định hướng Ứng dụng SBERT trong Tóm tắt Văn bản")
    add_p("Trong khuôn khổ Đồ án, hệ thống Tóm tắt văn bản vận hành theo cơ chế Bi-Encoder. SBERT không được dùng để so sánh trực tiếp cặp câu nào. Thay vào đó, giả sử bài báo có N câu, hệ thống sẽ bơm lần lượt N câu này vào SBERT để trích xuất ra N vector nhúng 768 chiều một cách hoàn toàn độc lập. Quá trình so sánh, tìm điểm tương đồng và phân nhóm ngữ nghĩa được đẩy hoàn toàn ra bên ngoài mạng Neural, giao cho thuật toán Phân cụm K-Means đảm nhiệm. Cách tiếp cận này giúp hệ thống tóm tắt tức thời chỉ trong vài mili-giây, vượt qua hoàn toàn nút thắt cổ chai về mặt hiệu năng của BERT.")

    add_heading_2("2.8. Ứng dụng SBERT trên Tiếng Việt: Vietnamese Bi-Encoder")
    add_p("Tiếng Việt là ngôn ngữ đơn lập, âm tiết rời rạc. Nếu dùng Tokenizer Tiếng Anh, từ \"Trí_tuệ_nhân_tạo\" sẽ bị xé nát. Do đó, hệ thống sử dụng Vietnamese Bi-Encoder (dựa trên PhoBERT) với bộ từ điển 64,000 từ vựng chuyên dụng, kết hợp thuật toán ghép từ (underthesea) để mã hóa chính xác thực thể tiếng Việt.")

    add_heading_2("2.9. Khung Chỉ số Đánh giá Kép (Dual-Evaluation Framework)")
    add_bullet("ROUGE (Recall-Oriented Understudy for Gisting Evaluation): ROUGE-1 (Unigram), ROUGE-2 (Bigram), ROUGE-L (LCS).", "Chỉ số Ngoại tại: ")
    add_bullet("BERTScore: Đánh giá bằng Deep Learning, bắt được hiện tượng từ đồng nghĩa mà ROUGE bỏ sót.", "Chỉ số Ngoại tại: ")
    add_bullet("Silhouette Score (đo độ sắc nét cụm K-Means) và Diversity Score (đo độ đa dạng chống lặp ý).", "Chỉ số Nội tại: ")

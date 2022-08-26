# Ứng dụng Học sâu phân loại Ung thư vú
-	EDA and Data Preprocessing: file này tôi dùng cho bước phân tích khai phá dữ liệu và tiền xử lý dữ liệu.
-	Train 5 models with truncated dataset: file này tôi dùng để huấn luyện 5 mô hình cơ sở với tập dữ liệu rút gọn (20% dữ liệu lớn).
-	Train EfficientNetB0 with all datasets: file này tôi dùng để huấn luyện mô hình EfficientNetB0 với toàn bộ tập dữ liệu gốc (dữ liệu lớn). EfficientNetB0 là kiến trúc mạng cho ra hiệu suất tốt nhất khi tôi đánh giá 5 mô hình cơ sở.
-	Test class weights with EfficientNetB0: file này tôi dùng để huấn luyện lại mô hình EfficientNetB0 với các class weights khác nhau.

# Giải thích các file
-	EDA and Data Preprocessing: file này tôi dùng cho bước phân tích khám phá dữ liệu và tiền xử lý dữ liệu.
-	Train 5 models with truncated dataset: file này tôi dùng để huấn luyện 5 mô hình cơ sở với tập dữ liệu rút gọn (20% dữ liệu lớn).
-	Train EfficientNetB0 with all datasets: file này tôi dùng để huấn luyện mô hình EfficientNetB0 với toàn bộ tập dữ liệu gốc (dữ liệu lớn). EfficientNetB0 là kiến trúc mạng cho ra hiệu suất tốt nhất khi tôi đánh giá 5 mô hình cơ sở.
-	Test class weights with EfficientNetB0: file này tôi dùng để huấn luyện lại mô hình EfficientNetB0 với các class weights khác nhau.
# Ứng dụng Học sâu phân loại Ung thư vú
## 1. Tóm tắt

Histopathological classification based on tissue samples and occurrences visual features in the images is an important stage in the breast cancer diagnostic process. This is a long and laborious process, and requires experience. This project, I worked within the field of Medical Imaging Diagnosis, tackling the classification of one of the major groups of cancer - breast cancer. Specifically, I applied Deep Learning technique in classifying benign and malignant invasive ductal carcinoma from histopathological images. I started out by performing Domain Research, and getting familiar with the domain I"m trying to solve a problem in. Then I proceeded with Exploratory Data Analysis, and began the standard Machine Learning Workflow. To solve the problem, I built CNN from scratch, as well as using predefined architectures, as well as use pre-defined architectures (such as the EfficientNetB0, VGG16, ResNet50, Xception). I identified the most promising base model as the EfficientNetB0, I performed hyperparameter tuning, and evaluated the model. The results show that the EfficientNetB0 model can correctly classify up to 88.03% with F1 score of 81.19%. Fast, accurate and early diagnosis improves the probability of survival. Accordingly, Machine Learning models can be deployed locally, and can process large sums of data in a fraction of the time it takes humans. As a result, doctors can focus on what they"re the best at - administering medicine, observing the effects and steering the procedure to help patients.

Phân loại mô bệnh học dựa trên các mẫu mô và sự xuất hiện các đặc điểm trực quan trong ảnh mô bệnh học là một giai đoạn quan trọng trong quy trình chẩn đoán Ung thư vú. Đây là quá trình lâu dài và tốn nhiều công sức, đồng thời đòi hỏi kinh nghiệm. Đồ án này tôi đã làm việc trong lĩnh vực chẩn đoán hình ảnh y tế, giải quyết việc phân loại một trong những nhóm chính của bệnh ung thư - Ung thư vú. Cụ thể, tôi đã áp dụng kỹ thuật Học sâu trong việc phân loại Ung thư biểu mô ống dẫn sữa xâm lấn lành tính và ác tính từ hình ảnh mô bệnh học. Tôi đã bắt đầu bằng cách thực hiện nghiên cứu và làm quen với lĩnh vực mà tôi đang cố gắng giải quyết vấn đề. Sau đó, tôi tiến hành Phân tích dữ liệu khám phá và bắt đầu đi vào quy trình làm việc của Học máy. Để giải quyết bài toán, tôi đã xây dựng CNN từ đầu, cũng như sử dụng các kiến trúc được xác định trước (chẳng hạn như EfficientNetB0, VGG16, ResNet50, Xception). Tôi xác định được mô hình cơ sở hứa hẹn nhất là EfficientNetB0, tôi đã thực hiện điều chỉnh siêu tham số và đánh giá mô hình. Kết quả cho thấy mô hình EfficientNetB0 có thể phân loại chính xác lên đến 88.03% với điểm F1 là 81.19%. Chẩn đoán nhanh, chính xác và sớm cải thiện xác suất sống sót. Theo đó, các mô hình Học máy có thể được triển khai ở địa phương và có thể xử lý lượng lớn dữ liệu trong một phần nhỏ thời gian mà con người cần. Kết quả là bác sĩ có thể tập trung vào những gì họ giỏi nhất - quản lý thuốc, quan sát các triệu chứng và đưa ra quy trình để giúp bệnh nhân.

## 2. Tổng quan
Trong dự án này, tôi muốn viết về ứng dụng phân loại của Deep learning (DL) ở cấp độ nghiên cứu!

Với tư cách là một kỹ sư máy tính trẻ - tôi đang khám phá triển vọng áp dụng của sự các thuật toán Machine learning (ML) cho các lĩnh vực khác nhau và trích xuất đặc trưng từ dữ liệu. Chẩn đoán ung thư sớm, chính xác và nhanh chóng giúp cải thiện xác suất sống sót, và chẩn đoán ung thư vú sớm có thể cứu sống tới 400.000 người mỗi năm. Các mô hình Học máy có thể được triển khai trên toàn cầu hoặc mỗi địa phương và có thể xử lý lượng lớn dữ liệu trong một phần nhỏ thời gian mà con người cần.

Ung thư biểu mô tuyến xâm lấn (IDC) là loại phụ phổ biến nhất của tất cả các loại ung thư vú. Ung thư vú là dạng ung thư phổ biến nhất ở phụ nữ. Xác định và phân loại chính xác các loại ung thư vú là một nhiệm vụ lâm sàng quan trọng và các phương pháp tự động có thể được sử dụng để tiết kiệm thời gian và giảm thiểu sai sót.

Là một người thực hành ML, bạn có thể giúp tạo ra sự khác biệt.

Trong dự án này, tôi sẽ làm việc trong lĩnh vực Chẩn đoán Hình ảnh Y tế, giải quyết việc phân loại một trong những nhóm chính của bệnh ung thư - Ung thư vú.

Dự án Phân loại ung thư vú với Keras và TensorFlow, tôi sẽ đi sâu vào một dự án thực hành, từ đầu đến cuối, xem xét thử thách là gì, phần thưởng sẽ là gì khi giải quyết nó. Cụ thể, tôi sẽ phân loại ung thư biểu mô tuyến xâm lấn lành tính và ác tính từ hình ảnh mô bệnh học. Nếu bạn không quen với thuật ngữ này - không cần lo lắng, nó đã được tôi đề cập trong dự án.

Tôi sẽ bắt đầu bằng cách thực hiện tìm hiểu và làm quen với Ung thư vú, đối tượng mà tôi đang cố gắng giải quyết vấn đề. Sau đó, tôi sẽ tiến hành Phân tích dữ liệu khám phá và bắt đầu Quy trình làm việc tiêu chuẩn của học máy. Đối với hướng dẫn này, chúng ta sẽ xây dựng CNN từ đầu , cũng như sử dụng các kiến trúc được xác định trước (chẳng hạn như họ EfficientNet hoặc họ ResNet). Khi tôi đánh giá được mô hình cơ sở hứa hẹn nhất, tôi sẽ thực hiện điều chỉnh siêu tham số và đánh giá mô hình.

## 3. Học máy trong y học
Học máy ngày càng được ứng dụng nhiều hơn trong y học và đang giúp cứu sống nhiều người khỏi nhiều tình trạng bệnh lý khác nhau. Ứng dụng của Học máy trong Y học là rất lớn và là một chủ đề cực kỳ phức tạp, nhưng một số lĩnh vực chính bao gồm:

Y học chính xác (Điều chỉnh thuốc cho bệnh nhân)
Chẩn đoán hình ảnh y tế (Chẩn đoán điều kiện dựa trên hình ảnh, v.v.)
Khám phá thuốc (Tạo ra các cấu trúc như protein hoặc các phân tử giống thuốc, dự đoán hoạt tính sinh học, v.v.)

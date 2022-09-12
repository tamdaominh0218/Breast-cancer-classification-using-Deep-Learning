# Giải thích các file
-	EDA and Data Preprocessing: file này tôi dùng cho bước phân tích khám phá dữ liệu và tiền xử lý dữ liệu.
-	Train 5 models with truncated dataset: file này tôi dùng để huấn luyện 5 mô hình cơ sở với tập dữ liệu rút gọn (20% dữ liệu lớn).
-	Train EfficientNetB0 with all datasets: file này tôi dùng để huấn luyện mô hình EfficientNetB0 với toàn bộ tập dữ liệu gốc (dữ liệu lớn). EfficientNetB0 là kiến trúc mạng cho ra hiệu suất tốt nhất khi tôi đánh giá 5 mô hình cơ sở.
-	Test class weights with EfficientNetB0: file này tôi dùng để huấn luyện lại mô hình EfficientNetB0 với các class weights khác nhau.
# Ứng dụng Học sâu phân loại Ung thư vú
## I. Tóm tắt

Histopathological classification based on tissue samples and occurrences visual features in the images is an important stage in the breast cancer diagnostic process. This is a long and laborious process, and requires experience. This project, I worked within the field of Medical Imaging Diagnosis, tackling the classification of one of the major groups of cancer - breast cancer. Specifically, I applied Deep Learning technique in classifying benign and malignant invasive ductal carcinoma from histopathological images. I started out by performing Domain Research, and getting familiar with the domain I"m trying to solve a problem in. Then I proceeded with Exploratory Data Analysis, and began the standard Machine Learning Workflow. To solve the problem, I built CNN from scratch, as well as using predefined architectures, as well as use pre-defined architectures (such as the EfficientNetB0, VGG16, ResNet50, Xception). I identified the most promising base model as the EfficientNetB0, I performed hyperparameter tuning, and evaluated the model. The results show that the EfficientNetB0 model can correctly classify up to 88.03% with F1 score of 81.19%. Fast, accurate and early diagnosis improves the probability of survival. Accordingly, Machine Learning models can be deployed locally, and can process large sums of data in a fraction of the time it takes humans. As a result, doctors can focus on what they"re the best at - administering medicine, observing the effects and steering the procedure to help patients.

Phân loại mô bệnh học dựa trên các mẫu mô và sự xuất hiện các đặc điểm trực quan trong ảnh mô bệnh học là một giai đoạn quan trọng trong quy trình chẩn đoán Ung thư vú. Đây là quá trình lâu dài và tốn nhiều công sức, đồng thời đòi hỏi kinh nghiệm. Đồ án này tôi đã làm việc trong lĩnh vực chẩn đoán hình ảnh y tế, giải quyết việc phân loại một trong những nhóm chính của bệnh ung thư - Ung thư vú. Cụ thể, tôi đã áp dụng kỹ thuật Học sâu trong việc phân loại Ung thư biểu mô ống dẫn sữa xâm lấn lành tính và ác tính từ hình ảnh mô bệnh học. Tôi đã bắt đầu bằng cách thực hiện nghiên cứu và làm quen với lĩnh vực mà tôi đang cố gắng giải quyết vấn đề. Sau đó, tôi tiến hành Phân tích dữ liệu khám phá và bắt đầu đi vào quy trình làm việc của Học máy. Để giải quyết bài toán, tôi đã xây dựng CNN từ đầu, cũng như sử dụng các kiến trúc được xác định trước (chẳng hạn như EfficientNetB0, VGG16, ResNet50, Xception). Tôi xác định được mô hình cơ sở hứa hẹn nhất là EfficientNetB0, tôi đã thực hiện điều chỉnh siêu tham số và đánh giá mô hình. Kết quả cho thấy mô hình EfficientNetB0 có thể phân loại chính xác lên đến 88.03% với điểm F1 là 81.19%. Chẩn đoán nhanh, chính xác và sớm cải thiện xác suất sống sót. Theo đó, các mô hình Học máy có thể được triển khai ở địa phương và có thể xử lý lượng lớn dữ liệu trong một phần nhỏ thời gian mà con người cần. Kết quả là bác sĩ có thể tập trung vào những gì họ giỏi nhất - quản lý thuốc, quan sát các triệu chứng và đưa ra quy trình để giúp bệnh nhân.

## II. Tổng quan
Trong dự án này, tôi muốn viết về ứng dụng phân loại của Deep learning (DL) ở cấp độ nghiên cứu!

Với tư cách là một kỹ sư máy tính trẻ - tôi đang khám phá triển vọng của sự ứng dụng các thuật toán Machine learning (ML) cho các lĩnh vực khác nhau và trích xuất đặc trưng từ dữ liệu. Chẩn đoán ung thư sớm, chính xác và nhanh chóng giúp cải thiện xác suất sống sót, và chẩn đoán ung thư vú sớm có thể cứu sống tới 400.000 người mỗi năm. Các mô hình Học máy có thể được triển khai trên toàn cầu hoặc mỗi địa phương và có thể xử lý lượng lớn dữ liệu trong một phần nhỏ thời gian mà con người cần.

Ung thư biểu mô tuyến xâm lấn (IDC) là loại phụ phổ biến nhất của tất cả các loại ung thư vú. Ung thư vú là dạng ung thư phổ biến nhất ở phụ nữ. Xác định và phân loại chính xác các loại ung thư vú là một nhiệm vụ lâm sàng quan trọng và các phương pháp tự động có thể được sử dụng để tiết kiệm thời gian và giảm thiểu sai sót.

Là một người thực hành ML, bạn có thể giúp tạo ra sự khác biệt.

Trong dự án này, tôi sẽ làm việc trong lĩnh vực Chẩn đoán Hình ảnh Y tế, giải quyết việc phân loại một trong những nhóm chính của bệnh ung thư - Ung thư vú.

Dự án Phân loại ung thư vú với Keras và TensorFlow, tôi sẽ đi sâu vào một dự án thực hành, từ đầu đến cuối, xem xét thử thách là gì, phần thưởng sẽ là gì khi giải quyết nó. Cụ thể, tôi sẽ phân loại ung thư biểu mô tuyến xâm lấn lành tính và ác tính từ hình ảnh mô bệnh học. Nếu bạn không quen với thuật ngữ này - không cần lo lắng, nó đã được tôi đề cập trong dự án.

Tôi sẽ bắt đầu bằng cách thực hiện tìm hiểu và làm quen với Ung thư vú, đối tượng mà tôi đang cố gắng giải quyết vấn đề. Sau đó, tôi sẽ tiến hành Phân tích dữ liệu khám phá và bắt đầu Quy trình làm việc tiêu chuẩn của học máy. Đối với hướng dẫn này, chúng ta sẽ xây dựng CNN từ đầu , cũng như sử dụng các kiến trúc được xác định trước (chẳng hạn như họ EfficientNet hoặc họ ResNet). Khi tôi đánh giá được mô hình cơ sở hứa hẹn nhất, tôi sẽ thực hiện điều chỉnh siêu tham số và đánh giá mô hình.

## III. Học máy trong y học
Học máy ngày càng được ứng dụng nhiều hơn trong y học và đang giúp cứu sống nhiều người khỏi nhiều tình trạng bệnh lý khác nhau. Ứng dụng của Học máy trong Y học là rất lớn và là một chủ đề cực kỳ phức tạp, nhưng một số lĩnh vực chính bao gồm:

- Y học chính xác (Điều chỉnh thuốc cho bệnh nhân)
- Chẩn đoán hình ảnh y tế (Chẩn đoán điều kiện dựa trên hình ảnh, v.v.)
- Khám phá thuốc (Tạo ra các cấu trúc như protein hoặc các phân tử giống thuốc, dự đoán hoạt tính sinh học, v.v.)

## 1. Tuyên bố thách thức/vấn đề
Hãy dành một chút thời gian để xác định vấn đề mà tôi đang cố gắng giải quyết:

Ung thư thường dễ nhận thấy trong mô và có thể dễ dàng điều trị hơn khi được phát hiện sớm. Mô học nghiên cứu các mô và Bệnh học nghiên cứu bệnh tật. Mô bệnh học nghiên cứu các bệnh trong mô! Các nhà giải phẫu bệnh kiểm tra hình ảnh của mô (hình ảnh mô học) và đưa ra kết luận. Ung thư giết chết 10 triệu người mỗi năm và là một trong những nguyên nhân gây tử vong hàng đầu trên toàn cầu. Cùng với ung thư phổi, ruột kết và dạ dày - ung thư vú giết chết 700.000 người mỗi năm. Một số khu vực có thể không có thiết bị hoặc phòng ý tế cần thiết để chẩn đoán trở thành một quy trình nhanh chóng, vì vậy bệnh nhân có thể phải đi lại khó khăn để được chẩn đoán, kéo dài thời gian họ không thể điều trị.

Theo một nghĩa nào đó, các nhà nghiên cứu bệnh học đang thực hiện phân loại (lành tính hoặc ác tính) dựa trên các mẫu và sự xuất hiện trong hình ảnh (các đặc điểm trực quan). Đây là một quá trình lâu dài và tốn nhiều công sức, đồng thời đòi hỏi kinh nghiệm.

![image](https://user-images.githubusercontent.com/110167646/186890301-86af4711-83e5-45fb-b861-cb0b4be595ab.png)

## 2. Phần thưởng cho việc giải quyết vấn đề
Chẩn đoán nhanh, chính xác và sớm cải thiện xác suất sống sót. Các mô hình Học máy có thể được triển khai trên toàn cầu hoặc cục bộ và có thể xử lý lượng lớn dữ liệu trong một phần nhỏ thời gian mà con người cần. Trên nhiều trường hợp khác nhau - người ta đã chứng minh rằng các mô hình Học máy, khi được huấn luyện đúng cách, có thể phân biệt các đặc trưng tốt hơn con người và có thể thực hiện phân loại hình ảnh ở mức độ chính xác cao hơn, ngay cả khi không có nhiều ngữ cảnh hoặc độ phân giải hình ảnh thấp.

Theo Cleveland Clinic :

Ung thư biểu mô ống xâm lấn có khả năng chữa khỏi khá cao, đặc biệt là khi được phát hiện và điều trị sớm. Tỷ lệ sống sót sau năm năm đối với ung thư biểu mô ống xâm lấn là cao - gần 100% khi được điều trị sớm. Nếu ung thư đã lan sang các mô khác trong khu vực, tỷ lệ sống sót sau năm năm là 86%. Nếu ung thư đã di căn đến các vùng xa của cơ thể, tỷ lệ sống sót sau 5 năm là 28%.

Một tính toán nhanh cho thấy việc phát hiện sớm có thể cứu sống 400.000 người mỗi năm. Có một động lực lớn để cung cấp các công cụ chẩn đoán toàn cầu, có thể truy cập, chính xác và nhanh chóng, đặc biệt là ở các khu vực khó có được chuyên môn. Với các nhiệm vụ dễ tự động hóa, bác sĩ có thể tập trung vào những gì họ giỏi nhất - quản lý thuốc, quan sát các triệu chứng và chỉ dẫn quy trình điều trị để giúp bệnh nhân.

![image](https://user-images.githubusercontent.com/110167646/186890958-9632d1d6-d5f9-4133-8bb0-0f9929bbc523.png)

## 3. Tìm hiểu và giới thiệu bệnh
Hãy dành một chút thời gian để làm quen với đối tượng mà tôi đang làm việc. Khi cố gắng giải quyết một vấn đề trong bất kỳ lĩnh vực nào, ít nhất bạn phải có kiến thức thô sơ về những gì bạn đang cố gắng giải quyết, tại sao bạn đang cố gắng giải quyết nó và dữ liệu có ý nghĩa gì trong ngữ cảnh của lĩnh vực. Nếu không biết bất kỳ điều gì về đối tượng - thật khó để biết liệu một mô hình có thực sự hoạt động hay không. Theo nguyên tắc chung - tốt nhất bạn nên tham khảo ý kiến của một người nào đó trong lĩnh vực này (1 bác sỹ chuyên khoa) và lấy ý kiến đóng góp của họ, đặc biệt là trong giai đoạn phát triển mô hình sau này.

Mặc dù vậy - để bắt đầu, bạn thường phải tự làm, vì vậy việc có thể nhanh chóng nắm được một số khái niệm cơ bản là rất quan trọng!

Ung thư biểu mô ống dẫn trứng xâm lấn (IDC) cho đến nay là loại phụ ung thư vú phổ biến nhất, chiếm 80% các trường hợp. Chỉ bằng cách giải quyết một loại phụ này, chúng tôi có thể giải quyết 80% các trường hợp.

Các khối u là những bó tế bào không được bó lại và phát triển thành những cục rắn. Các khối u có thể là lành tính (không phải ung thư) và tập trung vào một vùng cụ thể và có thể không gây ra bất kỳ vấn đề nào. Tuy nhiên, chúng có thể phát triển và gây ra các vấn đề thông qua kích thước tuyệt đối. Nếu một khối u bắt đầu phát triển bên ngoài vùng hỗn hợp của nhóm tế bào - nó sẽ trở thành ác tính (ung thư). Ung thư có thể xâm lấn mô cục bộ hoặc di căn và tấn công mô xa hơn. Còn nhiều điều cần nói về khối u và ung thư, bao gồm các dạng phụ và mức độ của nó, nhưng bộ dữ liệu mà chúng tôi đang làm việc chỉ đơn giản là phân loại hình ảnh là không ung thư (lành tính) và ung thư (ác tính).

Đối với tập dữ liệu cụ thể này, bạn cần phải có rất ít kiến thức y tế đáng ngạc nhiên để xây dựng một bộ phân loại có khả năng. Điều này một phần lớn là do các bác sĩ y học đã phải mất hàng trăm năm tích lũy kiến thức khoa học để ghi nhãn và chuẩn bị bộ dữ liệu mà từ đó chúng ta có thể suy ra kiến thức. Dựa trên kinh nghiệm và chuyên môn của họ, chúng tôi có thể xây dựng các mô hình để khai thác và phân loại đối tượng địa lý, với mức độ chính xác và tính toàn vẹn cao.

Đối với một kỹ sư Học máy - nhiệm vụ này gần như chỉ tập trung vào việc phân loại hình ảnh thông thường! Tuy nhiên, có một số tác động nhất định đi kèm với tập dữ liệu này, hiếm khi xuất hiện trong các tập dữ liệu khác mà bạn có thể đã làm việc trước đây. Tôi sẽ đặc biệt tập trung vào việc đưa ra các phỏng đoán có học thức trong phần sau, đồng thời xem xét sự mất cân bằng trong lớp học, sự gia tăng, học tập nhạy cảm với chi phí, v.v.

## IV. Phân tích dữ liệu khám phá (EDA) 
## 1. Loading the Data

Chúng tôi sẽ bắt đầu bằng cách tải xuống bộ dữ liệu và tải nó vào. Chúng tôi sẽ làm việc với bộ dữ liệu Hình ảnh [Mô bệnh học Vú](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images). Nó chứa 198738 bản vá hình ảnh IDC (-) và 78786 bản vá hình ảnh IDC (+) .

- IDC (-) đề cập đến các trường hợp lành tính
- IDC (+) dùng để chỉ các trường hợp ác tính

Lưu ý: IDC (-) trong tập dữ liệu này ngụ ý rằng bệnh nhân không bị ung thư biểu mô ống dẫn sữa xâm lấn . Nó ngụ ý rằng họ có một trường hợp lành tính hoặc mô bình thường, chứ không phải là một trường hợp ác tính . Bên cạnh IDC, một tình trạng khác cũng tồn tại - Ung thư biểu mô ống không xâm lấn còn được gọi là ung thư biểu mô tại chỗ (DCIS) .

Bộ dữ liệu lấy từ một nghiên cứu năm 2016 - ["Học sâu để phân tích hình ảnh bệnh lý kỹ thuật số: Hướng dẫn toàn diện với các trường hợp sử dụng được chọn"](https://pubmed.ncbi.nlm.nih.gov/27563488/) của Andrew Janowczyk và Anant Madabhushi. Nghiên cứu của họ tập trung vào một số nhiệm vụ, một trong số đó là xác định IDC, mà họ có điểm F1 là 0,7648 trên 50k bản vá thử nghiệm.

Tập dữ liệu mà chúng tôi đang làm việc được lấy từ 279 bệnh nhân, mỗi bệnh nhân có một ID duy nhất. Mỗi bệnh nhân có một thư mục chuyên dụng, được đặt tên theo ID của họ, với hai thư mục con 0 và 1. Thư mục được đặt tên 0 bao gồm hình ảnh của các mẫu mô lành tính (những mẫu không có dấu IDC). Thư mục được đặt tên 1 bao gồm hình ảnh của các mẫu mô ác tính (những mẫu có chứa dấu IDC).

Hình ảnh mô bệnh học có kích thước lớn, còn các đặc điểm và điểm đánh dấu rất nhỏ, đó là lý do tại sao hình ảnh được chia nhỏ thành các mảng , kích thước 50x50 pixel. Do đó, mỗi bệnh nhân có nhiều bản vá hình ảnh, gộp lại sẽ bao gồm toàn bộ hình ảnh.

Mỗi bản vá có một định dạng tên riêng - uxXyYclassC.png, trong đó u là ID của bệnh nhân, x là tọa độ X mà từ đó bản vá được trích xuất, ylà tọa độ Y mà từ đó bản vá được trích xuất và class là 0 hoặc 1, biểu thị liệu có các dấu IDC hay không hoặc không có trong bản vá đó.

Các tọa độ được đưa ra để toàn bộ hình ảnh có thể được tái tạo từ các mảng, nhưng cũng để chúng ta có thể tô màu các mảng trong toàn bộ ảnh, đây là quy trình khá chuẩn trong bệnh học. Tôi sẽ sớm thực hiện việc này trong phần EDA!

Đó là tất cả những gì đang được nói - tôi đã tải xuống tệp zip và giải nén nó thành `breast-histopathology-images`:

```
data = os.listdir("./breast-histopathology-images/")
len(data)
# 279
```

Thật vậy, có 279 thư mục, biểu thị 279 ID bệnh nhân:
```
data[:10]
```

Chúng ta hãy xem xét 10 bệnh nhân đầu tiên:
```
['10253',
 '10254',
 '10255',
 '10256',
 '10257',
 '10258',
 '10259',
 '10260',
 '10261',
 '10262']
 ```
 
 Trong mỗi thư mục này, có một thư mục 0 và 1:
 ```
 patient_10253 = os.listdir("./breast-histopathology-images/10253")
# ['0', '1']
```

Và trong mỗi chúng, một số hình ảnh:
```
patient_10253_0 = os.listdir("./breast-histopathology-images/10253/0")
patient_10253_1 = os.listdir("./breast-histopathology-images/10253/1")

print(patient_10253_0[:5])
# ['10253_idx5_x1001_y1001_class0.png', '10253_idx5_x1001_y1051_class0.png', '10253_idx5_x1001_y1101_class0.png', '10253_idx5_x1001_y1151_class0.png', '10253_idx5_x1001_y1201_class0.png']
print(patient_10253_1[:5])
# ['10253_idx5_x501_y351_class1.png', '10253_idx5_x501_y401_class1.png', '10253_idx5_x551_y301_class1.png', '10253_idx5_x551_y351_class1.png', '10253_idx5_x551_y401_class1.png']
```

Các tên tệp ở cuối dài hơn, nhưng chứa dữ liệu thực sự có giá trị! Làm việc với danh sách không thực sự lý tưởng, vì vậy chúng ta hãy chỉ lấy dữ liệu cho bệnh nhân đầu tiên và lưu trữ nó trong một vài `DataFrames` thay vào đó, kết hợp chúng với nhau thành một dữ liệu duy nhất:
```
df_0 = pd.DataFrame()

for path in patient_10253_0:
    split = path.split('_')
    # Extract elements 2 and 3, substringing the first char
    x_coord = split[2][1:]
    y_coord = split[3][1:]
    idc_class = 0
    
    data = {"path":"./breast-histopathology-images/10253/0/"+path,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "idc_class": idc_class}
    
    df_0 = df_0.append(data, ignore_index=True)
    
    
print(df_0)

df_1 = pd.DataFrame()

for path in patient_10253_1:
    split = path.split('_')
    # Extract elements 2 and 3, substringing the first char
    x_coord = split[2][1:]
    y_coord = split[3][1:]
    idc_class = 1
    # Hardcoded path for now, we'll address this later
    data = {"path":"./breast-histopathology-images/10253/1/"+path,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "idc_class": idc_class}
    
    df_1 = df_1.append(data, ignore_index=True)
    
    
# Combine dataframes
df = df_0.append(df_1).reset_index()
# Convert the coordinates to integers, from objects
df['x_coord'] = df['x_coord'].astype('int')
df['y_coord'] = df['y_coord'].astype('int')
```

Khung dữ liệu này hiện bao gồm:

![image](https://user-images.githubusercontent.com/110167646/186894894-02fd91f0-82bf-4226-812e-4ea7d3095937.png)

Tuyệt vời, tôi có các đường dẫn vá (có nghĩa là, tôi có thể lấy hình ảnh cho từng cái), x_coordvà y_coordcho từng bản vá (nghĩa là, chúng tôi biết vị trí của nó trong hình ảnh) và idc_class cho bản vá cụ thể đó! Sử dụng dữ liệu này, chúng tôi có thể tái tạo lại toàn bộ hình ảnh ban đầu và thậm chí còn đánh dấu các phần tích cực IDC trong chúng, với một màu khác như thể bằng bút đánh dấu.

## 2. Phân tích dữ liệu khám phá (EDA)
Trước khi tạo lại hình ảnh, hãy thử tạo một biểu đồ phân tán với dữ liệu này, đặt `c` (màu) tương ứng với bắt đầu của các bản vá (góc trên bên trái):`idc_class`
```
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x = df['x_coord'], y=df['y_coord'], c=df['idc_class'], cmap='coolwarm')
plt.show()
```

Điều này dẫn đến một hình ảnh khá rõ ràng:

![image](https://user-images.githubusercontent.com/110167646/186895421-7d13e662-53c7-43ef-a192-ec22f44ec58f.png)

Có vẻ như một số bản vá bị thiếu! Trang Kaggle không đề cập đến vấn đề này, nhưng nhìn chung, các điểm dữ liệu thường bị thiếu do bị cắt bớt trong quá trình xử lý dữ liệu thô hoặc do một công cụ bị lỗi, khiến điểm dữ liệu mất giá trị thông tin.

Mặc dù vậy, chúng ta vẫn sẽ ổn nếu không có một số bản vá lỗi này. Hãy tiếp tục và thử tái tạo lại hình ảnh này, vẽ một bản vá thích hợp , thay vì một điểm đánh dấu biểu đồ phân tán. Vì chúng tôi đang xử lý một danh sách các bản vá phong phú và việc phải đối phó với nhiều Axestrường hợp có thể khá mệt mỏi - hãy tạo một lưới các "bản vá" (stand-in) và mặt nạ cho các bản vá tích cực IDC, có thể được tô màu bằng một màu khác và được "đóng dấu trên" hình ảnh bên dưới. Phương pháp này được lấy cảm hứng từ Kaggle Grandmaster Laura Fink!

Lưới và mặt nạ đều sẽ bắt đầu với các giá trị đứng, lý tưởng nhất là màu trắng. Màu trắng được biểu thị dưới dạng tất cả các kênh RGB, với các giá trị là 255, vì vậy chúng tôi có thể tạo lưới và mặt nạ màu trắng với:
```
grid = 255*np.ones(shape = (100, 100, 3)).astype(np.uint8)
mask = 255*np.ones(shape = (100, 100, 3)).astype(np.uint8)
```

Bây giờ cả hai về cơ bản sẽ là:
```
array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]],
        ...
```
Và hình dạng:
```
print(grid.shape) # (100, 100, 3)
```
        
Nếu chúng ta hình dung chúng dưới dạng hình ảnh - sẽ không có nhiều điều xảy ra - chúng đều là các pixel màu trắng. Hãy sử dụng các giá trị tối đa của chúng tôi x_coordvà của chúng tôi y_coordlàm trình bao bọc để biết có bao nhiêu pixel trong hình ảnh (hình dạng của lưới và mặt nạ) và đối với mỗi hình ảnh trong DataFrame, lấy x_coordvà y_coord, đặt ranh giới của nó (__coord+50) và in dấu pixel giá trị của hình ảnh trên lưới:
```
max_x = df['x_coord'].max()
max_y = df['y_coord'].max()

# Shape of (2101, 2651, 3)
# A placeholder for each pixel, with 3 color options and max values for each channel (RGB)
grid = 255*np.ones(shape = (max_y + 50, max_x + 50, 3)).astype(np.uint8)
mask = 255*np.ones(shape = (max_y + 50, max_x + 50, 3)).astype(np.uint8)

for i in range(len(df)):
        # Get image and label
        image = cv2.imread(df['path'][i])
        idc_class = df['idc_class'][i]
            
        # Extract X and Y coordinates
        x_coord = df['x_coord'][i]
        y_coord = df['y_coord'][i]
        # Add 50 pixels to find ending boundary for each image
        x_end = x_coord + 50
        y_end = y_coord + 50
        
        # Assign image pixel values to placeholder 255 values
        """
        Image is something along the lines of:
        [[[206 164 226]
          [196 154 224]
          [211 175 225]
          ...
          [237 221 240]
          [214 184 232]
          [235 213 243]],
          ...        
        """
        # `grid` will then contain each patch's image values encoded into the grid
        grid[y_coord:y_end, x_coord:x_end] = image
        
        # If `idc_class` is `1`, change the RED channel of the `mask` to 255 (intense red)
        # and other channels to `0` (remove color info, leaving just red)
        if idc_class == 1:
            mask[y_coord:y_end, x_coord:x_end, :1] = 255
            mask[y_coord:y_end, x_coord:x_end, 1:] = 0
```

Cuối cùng, chúng ta có thể hiển thị lưới và mặt nạ ở trên nó:
```
plt.figure(figsize=(12, 12))
plt.imshow(grid)
plt.imshow(mask, alpha=0.1)
plt.show()
```

![image](https://user-images.githubusercontent.com/110167646/186896091-de2bd394-e0f5-495c-af36-e97f3a427cfa.png)

Hãy tải thêm dữ liệu, tạo ra một DataFramebao gồm thông tin của tất cả các bản vá, ID bệnh nhân, lớp IDC của họ, v.v. và thống kê tóm tắt biểu đồ sẽ giúp chúng ta có cái nhìn chung về tập dữ liệu:
```
from glob import glob
data = glob('./breast-histopathology-images/**/*.png', recursive=True)
print(len(data))
# 277524

dfs = []

for path in data:
    split = path.split('_')
    # Extract elements 2 and 3, substringing the first char
    patient_id = split[0].split('\\')[1]
    x_coord = split[2][1:]
    y_coord = split[3][1:]
    idc_class = split[4][-5]
    
    df_data = {"patient_id": patient_id,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "idc_class": idc_class,
            "path": path}
    df = pd.DataFrame()
    dfs.append(df.append(df_data, ignore_index=True))
    
df_all = pd.concat(dfs)
# Reset index aftet concatenation
df_all = df_all.reset_index(drop=True)

df_all['x_coord'] = df_all['x_coord'].astype('int')
df_all['y_coord'] = df_all['y_coord'].astype('int')
df_all['idc_class'] = df_all['idc_class'].astype('int')
```

Lưu ý: Đoạn mã này mất một chút thời gian để thực thi, vì chúng tôi đang tạo một đoạn mã khá lớn DataFrame.

Hãy bắt đầu với một âm mưu đếm của idc_class:
```
import seaborn as sns
sns.countplot(x='idc_class', data=df_all)
```

![image](https://user-images.githubusercontent.com/110167646/186896251-f59703f3-2ddf-4993-943e-fb501cfcaf5e.png)

Có một sự mất cân bằng khá lớn giữa các lớp. Điều này sẽ làm cho việc khái quát hóa khó khăn hơn, vì việc tập trung vào tầng lớp thống trị có thể tỏ ra thuận lợi với một số kiến ​​trúc nhất định. Chúng ta sẽ xem xét chi tiết về sự mất cân bằng lớp và ý nghĩa của nó trong bài học tiếp theo, trong quá trình xử lý trước dữ liệu.

Hãy xem tần suất hình ảnh trên mỗi bệnh nhân - chúng có trải đều không?
```
df_all['patient_id'].value_counts()
```

Một số bệnh nhân có nhiều dữ liệu hơn những bệnh nhân khác và số lượng dữ liệu trên mỗi bệnh nhân rất đa dạng!
```
13693    2395
16550    2302
10288    2278
10308    2278
9323     2216
         ... 
16895     151
9175      118
8957      111
9262       94
16534      63
```

Điều này sẽ không làm cho việc đào tạo và học hỏi từ các bản vá này trở nên khó khăn hơn - chúng tôi sẽ đào tạo một hệ thống chẩn đoán IDC bằng bản vá , không phải bằng toàn bộ hình ảnh. Mặc dù vậy, sẽ khó hơn một chút nếu chúng tôi muốn chạy thử nghiệm cho tất cả bệnh nhân và làm nổi bật các vùng trong mô được đánh giá là dương tính với IDC, vì không phải tất cả bệnh nhân đều có toàn bộ hình ảnh có thể được sử dụng cho chú thích.

Hãy vẽ toàn bộ hình ảnh cho một số bệnh nhân từ đầu danh sách:
```
patient_ids = ['13693', '16550', '10288', '10308', '9323']

for patient_id in patient_ids:
    df = df_all.loc[df_all['patient_id'] == patient_id].reset_index(drop=True)
    max_x = df['x_coord'].max()
    max_y = df['y_coord'].max()
    
    grid = 255*np.ones(shape = (max_y + 50, max_x + 50, 3)).astype(np.uint8)
    mask = 255*np.ones(shape = (max_y + 50, max_x + 50, 3)).astype(np.uint8)

    for i in range(len(df)):
        # Get image and label
        image = cv2.imread(df['path'][i])
        # Image shape might not be 50x50, in which case, it's a broken patch
        # and we don't want to load it in
        if(image.shape==(50, 50, 3)):
            idc_class = df['idc_class'][i]
            x_coord = df['x_coord'][i]
            y_coord = df['y_coord'][i]
            x_end = x_coord + 50
            y_end = y_coord + 50
            
            grid[y_coord:y_end, x_coord:x_end] = image

            if idc_class == 1:
                mask[y_coord:y_end, x_coord:x_end, :1] = 255
                mask[y_coord:y_end, x_coord:x_end, 1:] = 0
                
    plt.figure(figsize=(8, 8))
    plt.suptitle(f'Patient ID: {patient_id}')
    plt.imshow(grid)
    plt.imshow(mask, alpha=0.2)
    plt.show()
```

![image](https://user-images.githubusercontent.com/110167646/186896400-582b2f1a-70fa-40c2-acef-b585916f6621.png)

Một số trang chiếu có diện tích nhỏ được che bởi mặt nạ, chẳng hạn như với bệnh nhân 10288 , mặc dù một số có diện tích khá lớn, chẳng hạn như bệnh nhân 10308 . Hãy xem liệu chúng ta có thể đào tạo một bộ phân loại để tìm ra lý do tại sao một số bản vá lỗi IDC tích cực và bản vá lỗi nào không.

## V. Quy trình làm việc của Học máy
`Đang cập nhật...`
## 1. Tiền xử lý dữ liệu
## 2. Vấn đề mất cân bằng giữa 2 lớp
## 3. Huấn luyện mô hình - CNN từ đầu
## 4. Huấn luyện mô hình - EfficientNetB0
## 5. Huấn luyện mô hình - VGG16
## 6. Huấn luyện mô hình - ResNet50
## 7. Huấn luyện mô hình - Xception
## 8. Điều chỉnh siêu tham số
## 9. Các mô hình hiệu quả khác
## 10. Huấn luyện mô hình đã chọn trên tất cả dữ liệu
## VI. Kết luận

`Đang cập nhật...`

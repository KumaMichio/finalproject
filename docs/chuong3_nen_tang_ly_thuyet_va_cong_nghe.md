# CHƯƠNG 3. NỀN TẢNG LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG

Chương 2 đã xác định các yêu cầu chức năng và phi chức năng của hệ thống: phát
hiện và theo dõi phương tiện trong từng luồng video (UC1), phát hiện vi phạm
giao thông gồm vượt đèn đỏ, vượt vạch dừng và đi sai làn (UC2), dự đoán hướng
di chuyển của phương tiện và trực quan hoá trên bản đồ (UC3), cùng khả năng cấu
hình hệ thống (UC4); kèm theo các yêu cầu phi chức năng về xử lý thời gian thực
nhiều luồng camera (~15–20 FPS), tính minh bạch của kết quả và khả năng triển
khai trực tiếp trên dữ liệu CCTV mô phỏng. Chương 3 trình bày các nền tảng lý
thuyết và công nghệ được lựa chọn để hiện thực hoá từng yêu cầu trên, kèm theo
lý do lựa chọn và các hướng tiếp cận thay thế đã được cân nhắc. Nội dung được
trình bày theo đúng thứ tự luồng xử lý: phát hiện đối tượng (phần 3.1), kiến
trúc YOLO11 (phần 3.2), theo dõi đa đối tượng trong một camera (phần 3.3), phát
hiện vi phạm giao thông (phần 3.4), dự đoán hướng di chuyển và suy diễn hành
trình (phần 3.5) và kiến trúc dịch vụ thời gian thực – trực quan hoá (phần 3.6).

## 3.1 Phát hiện đối tượng

Bài toán phát hiện đối tượng (object detection) yêu cầu đồng thời xác định vị
trí (qua khung bao – bounding box) và phân loại các đối tượng xuất hiện trong
ảnh. Trong hệ thống, bài toán này phục vụ trực tiếp UC1 và là nền tảng cho mọi
mô-đun phía sau: theo dõi, phát hiện vi phạm và dự đoán hướng đi đều chỉ hoạt
động được trên đầu ra của bước phát hiện. Đồng thời, tốc độ xử lý ở bước này
quyết định phần lớn khả năng đáp ứng yêu cầu thời gian thực trên nhiều luồng
camera (mục 2.4). Các hướng tiếp cận phổ biến hiện nay gồm:

- **Mask R-CNN và họ hai giai đoạn (two-stage)** [1]: một mạng đề xuất vùng
  (region proposal) chạy trước một mạng phân loại/định vị tinh. Độ chính xác
  cao nhưng tốc độ suy luận chậm (khoảng vài khung hình/giây trên GPU máy tính),
  không khả thi cho xử lý nhiều luồng camera đồng thời theo thời gian thực.
- **SSD và EfficientDet (one-stage)** [4]: dự đoán trực tiếp khung bao và lớp
  trong một lượt duyệt mạng, nhanh hơn rõ rệt. Tuy nhiên độ chính xác trên các
  đối tượng nhỏ thường kém hơn — một hạn chế đáng kể với xe máy quan sát từ xa
  qua camera CCTV.
- **Họ mô hình YOLO (You Only Look Once)** [2]: cũng thuộc nhóm một giai đoạn,
  nhưng liên tục được cải tiến qua các phiên bản để thu hẹp khoảng cách độ chính
  xác với mô hình hai giai đoạn trong khi vẫn giữ tốc độ suy luận cao.

Đề tài lựa chọn **YOLO11** (Ultralytics, 2024) [3], cụ thể là biến thể nhỏ
(*small*), cho mô-đun phát hiện, vì ba lý do. Thứ nhất, tốc độ suy luận đủ nhanh
để xử lý nhiều luồng camera trên phần cứng thông thường mà không yêu cầu máy chủ
GPU chuyên dụng — đáp ứng trực tiếp yêu cầu hiệu năng và khả năng triển khai
(mục 2.4). Thứ hai, hệ sinh thái Ultralytics cung cấp sẵn công cụ huấn luyện lại
(fine-tuning) trên bộ nhãn tuỳ chỉnh, cho phép huấn luyện mô hình theo đúng các
lớp đối tượng đặc trưng giao thông Việt Nam (người, xe máy, ô tô, xe buýt, xe
tải) thay vì dùng nguyên bộ nhãn COCO gốc [23]. Thứ ba, so với các phiên bản
YOLO trước (YOLOv5/v8), YOLO11 cải thiện độ chính xác trên vật thể nhỏ mà không
tăng đáng kể độ trễ suy luận — phù hợp với đặc thù xe máy chiếm tỷ trọng lớn
nhưng kích thước nhỏ trong ảnh CCTV.

## 3.2 Kiến trúc YOLO11

YOLO11 có kiến trúc ba phần chính: backbone, neck và head [3].

**Backbone** đảm nhận việc trích xuất đặc trưng từ ảnh đầu vào, sử dụng các khối
tích chập kiểu Cross Stage Partial kết hợp gộp đa tỉ lệ (SPPF – Spatial Pyramid
Pooling Fast) để trích đặc trưng ở nhiều mức độ phân giải. So với YOLOv8,
backbone của YOLO11 ít tham số hơn nhưng khả năng biểu diễn gần như tương đương.

**Neck** tổng hợp đặc trưng từ các tầng khác nhau của backbone theo kiến trúc
kim tự tháp đặc trưng (FPN – Feature Pyramid Network) [5] kết hợp mạng tổng hợp
theo đường dẫn (PAN – Path Aggregation Network) [6]. Sự kết hợp này giữ được cả
đặc trưng ngữ nghĩa cấp cao (từ tầng sâu) lẫn đặc trưng không gian cấp thấp (từ
tầng nông), nhờ vậy mô hình bắt được đối tượng ở nhiều kích thước — đặc biệt
quan trọng để cùng lúc phát hiện xe máy nhỏ và xa lẫn ô tô/xe buýt lớn trong
một khung hình CCTV.

**Head** dùng kiến trúc *anchor-free*, dự đoán trực tiếp toạ độ khung bao, xác
suất lớp và điểm tin cậy cho từng đối tượng mà không cần các hộp neo cố định
được thiết kế trước.

Đề tài sử dụng biến thể *small* của YOLO11 với kích thước đầu vào 640×640 pixel —
một cấu hình cân bằng tốt giữa độ chính xác và tốc độ suy luận khi phải chạy
song song nhiều luồng camera.

### 3.2.1 Học chuyển giao và huấn luyện trên dữ liệu giao thông Việt Nam

Học chuyển giao (transfer learning) là kỹ thuật tận dụng tri thức từ một mô hình
đã được huấn luyện trên tập dữ liệu lớn (pretrained model) cho một bài toán mới
với dữ liệu nhỏ hơn [7], giúp rút ngắn thời gian huấn luyện và bù lại sự thiếu
hụt dữ liệu đặc thù. Đề tài khởi tạo mô hình từ trọng số đã huấn luyện trước
trên bộ dữ liệu COCO [23], sau đó tinh chỉnh (fine-tune) trên bộ dữ liệu giao
thông Việt Nam tự thu thập và gán nhãn từ video CCTV thực tế. Bộ nhãn gốc của
COCO được hợp nhất lại về các lớp đối tượng giao thông cần thiết (người, xe máy,
ô tô, xe buýt, xe tải), giúp mô hình tập trung học đúng các lớp xuất hiện trong
bối cảnh sử dụng thay vì toàn bộ 80 lớp tổng quát của COCO. Chiến lược này cho
phép mô hình thích nghi với đặc thù giao thông địa phương (mật độ xe máy cao,
góc nhìn từ trên cao của camera CCTV) mà không đánh mất khả năng nhận diện đã
học được từ tập dữ liệu lớn.

### 3.2.2 Hậu xử lý đầu ra

Đầu ra thô của YOLO11 là một tập hợp lớn các khung bao ứng viên kèm điểm tin cậy
và xác suất lớp; hậu xử lý gồm hai bước để chuyển tập này thành danh sách phát
hiện sạch.

Bước thứ nhất là lọc theo ngưỡng tin cậy (confidence threshold). Hệ thống dùng
một ngưỡng tin cậy tương đối thấp (mặc định 0,35) khi triển khai trên dữ liệu
CCTV, xuất phát từ đặc thù bài toán: trong giám sát giao thông, việc bỏ sót một
phương tiện gây ra hậu quả nghiêm trọng hơn (mất dấu, bỏ lọt vi phạm) so với một
lần phát hiện nhầm vốn dễ bị các bước theo dõi và lọc thời gian phía sau loại
bỏ. Để giảm chi phí bộ nhớ khi chạy nhiều luồng đồng thời, mô hình được suy luận
ở độ chính xác bán phần (FP16) khi phần cứng hỗ trợ.

Bước thứ hai là khử trùng lặp bằng Non-Maximum Suppression (NMS): hai khung bao
có độ chồng lấp (IoU – Intersection over Union) vượt ngưỡng sẽ được gộp lại, giữ
lại phát hiện có điểm tin cậy cao hơn, nhằm tránh việc một vật thể bị gán nhiều
khung bao chồng nhau.

## 3.3 Theo dõi đa đối tượng trong một camera

Sau khi có danh sách phát hiện ở mỗi khung hình, hệ thống cần liên kết các phát
hiện đó qua các khung hình liên tiếp để xác định "đây vẫn là cùng một phương
tiện" và gán cho nó một định danh ổn định trong phạm vi một luồng video — đây
chính là phần còn lại của UC1. Bài toán theo dõi đa đối tượng (Multi-Object
Tracking – MOT) thường theo mô hình *tracking-by-detection*: dùng một bộ lọc
chuyển động (thường là bộ lọc Kalman [8]) để dự đoán vị trí tiếp theo của các
đối tượng đang theo dõi, rồi ghép cặp các phát hiện mới với các dự đoán đó bằng
thuật toán ghép cặp tối ưu hai phía (Hungarian matching). Các hướng tiếp cận
khác nhau chủ yếu ở chỗ có dùng thêm đặc trưng hình ảnh (appearance) để ghép cặp
hay không:

- **SORT**: chỉ dùng độ chồng lấp khung bao (IoU) và bộ lọc Kalman; đơn giản,
  nhanh, nhưng dễ mất dấu khi đối tượng bị che khuất.
- **DeepSORT / StrongSORT**: bổ sung một mạng trích đặc trưng hình ảnh riêng cho
  từng đối tượng ở *mỗi khung hình* để ghép cặp chính xác hơn khi chuyển động
  không đủ tin cậy, nhưng trả giá bằng chi phí tính toán tăng đáng kể — dư thừa
  với một hệ thống cần theo dõi nhiều đối tượng trên nhiều luồng đồng thời, trong
  khi yêu cầu chỉ là duy trì định danh trong phạm vi một luồng video.
- **ByteTrack** [9]: thay vì chỉ giữ các phát hiện có độ tin cậy cao, ByteTrack
  giữ và ghép cặp *toàn bộ* khung bao phát hiện được, kể cả các khung bao có độ
  tin cậy thấp (chiến lược BYTE), dựa trên lập luận rằng một phát hiện điểm tin
  cậy thấp do bị che khuất một phần vẫn là tín hiệu hữu ích để duy trì track,
  chỉ cần không dùng nó để khởi tạo track mới.

Đề tài lựa chọn **ByteTrack**, triển khai qua thư viện `boxmot` [10]. Việc giữ
lại các phát hiện độ tin cậy thấp đặc biệt phù hợp với giao thông Việt Nam, nơi
xe máy di chuyển sát nhau và thường xuyên che khuất lẫn nhau một phần, khiến độ
tin cậy phát hiện dao động liên tục — nếu loại bỏ các phát hiện độ tin cậy thấp
như SORT, track của xe máy bị che khuất tạm thời sẽ dễ bị ngắt quãng và sinh ra
định danh mới không cần thiết. Đồng thời, vì ByteTrack chỉ dựa vào chuyển động
(không cần một mạng trích đặc trưng hình ảnh riêng), chi phí tính toán ở bước
theo dõi được giữ tối thiểu, dành tài nguyên cho các bước phát hiện và dự đoán.
Các tham số chính được cấu hình gồm: ngưỡng kích hoạt track 0,25, ngưỡng ghép
cặp 0,8, bộ đệm giữ track đã mất 30 khung hình, và tốc độ khung hình đặt riêng
theo từng camera.

## 3.4 Phát hiện vi phạm giao thông

UC2 yêu cầu hệ thống tự động xác định ba loại vi phạm — vượt đèn đỏ, vượt vạch
dừng và đi sai làn — từ dữ liệu track đầu ra của các bước trên. Có hai hướng
tiếp cận chính cho bài toán phát hiện hành vi bất thường/vi phạm trong giám sát
giao thông [11]:

- **Học máy không giám sát (unsupervised anomaly detection)**: huấn luyện một mô
  hình (ví dụ Isolation Forest [12] hoặc bộ tự mã hoá – autoencoder) để tự học
  "thế nào là hành vi bình thường" từ một lượng lớn dữ liệu track, rồi đánh dấu
  bất thường là những mẫu sai khác nhiều so với phân phối đã học.
- **Hệ luật suy diễn (rule-based reasoning)**: định nghĩa trực tiếp các điều
  kiện hình học và động học cụ thể (vị trí so với vạch dừng/làn đường, hướng di
  chuyển, vận tốc, trạng thái đèn) ứng với từng loại vi phạm cần phát hiện.

Đề tài lựa chọn **hệ luật suy diễn**. Lý do chính nằm ở yêu cầu minh bạch của kết
quả (mục 2.4): với hướng học máy không giám sát, hệ thống chỉ có thể báo "hành vi
này khác thường" mà khó giải thích vì sao, gây khó khăn cho người vận hành khi
cần hiểu căn cứ của một cảnh báo; trong khi với hệ luật, mỗi vi phạm gắn liền với
một điều kiện cụ thể, có thể diễn giải trực tiếp ("phương tiện vượt qua vạch dừng
trong khi đèn đang đỏ") và dễ điều chỉnh ngưỡng theo từng camera/tuyến đường mà
không cần huấn luyện lại. Ngoài ra, hướng học máy không giám sát đòi hỏi một
lượng dữ liệu "bình thường" đủ lớn và đặc thù cho từng địa điểm, trong khi vi
phạm thực tế vốn hiếm và khó thu thập đủ để kiểm chứng mô hình một cách tin cậy.
Hướng học máy không giám sát được ghi nhận là một hướng nâng cấp khả thi trong
tương lai (Chương 6).

### 3.4.1 Xác định trạng thái đèn tín hiệu

Điều kiện tiên quyết để phát hiện vi phạm vượt đèn đỏ là biết được trạng thái
đèn tín hiệu tại mỗi thời điểm. Đề tài xác định trạng thái đèn bằng phương pháp
phân tích màu trực tiếp trên khung hình: người vận hành đánh dấu trước toạ độ vị
trí của từng bóng đèn (đỏ/vàng/xanh) trong khung hình của mỗi camera (một phần
của UC4 – cấu hình hệ thống); tại mỗi khung hình, hệ thống trích một vùng nhỏ
quanh các toạ độ đó và so sánh độ sáng/sắc màu trong không gian màu HSV để xác
định đèn nào đang bật. Để chống nhiễu do dao động ánh sáng tức thời, trạng thái
đèn được làm trơn bằng cơ chế bỏ phiếu đa số trên một cửa sổ vài khung hình liên
tiếp. Trong trường hợp bóng đèn không nằm trong tầm nhìn của camera, hệ thống cho
phép khai báo trước **chu kỳ đèn tín hiệu** (thời lượng các pha) trong cấu hình
UC4 và suy ra trạng thái đèn theo thời gian. Cách tiếp cận khai báo toạ
độ/chu kỳ theo từng camera được chọn thay cho một mô hình học sâu nhận dạng đèn
vì nó không cần dữ liệu huấn luyện riêng, hoạt động ngay khi triển khai trên một
camera mới và minh bạch với người vận hành.

### 3.4.2 Hệ luật phát hiện vi phạm

Trên cơ sở trạng thái đèn và các vùng quan tâm (ROI) do người vận hành cấu hình,
ba loại vi phạm được phát hiện như sau:

- **Vượt đèn đỏ / vượt vạch dừng**: người vận hành khai báo vị trí vạch dừng và
  vùng giao lộ tương ứng. Khi đèn đang ở trạng thái đỏ, nếu một phương tiện đang
  di chuyển (vận tốc vượt ngưỡng tối thiểu) vượt qua vạch dừng tiến vào vùng giao
  lộ, hệ thống ghi nhận vi phạm. Ngưỡng vận tốc tối thiểu giúp loại bỏ các phương
  tiện đang dừng chờ đèn sát vạch.
- **Đi sai làn**: mỗi làn được khai báo bằng một vùng đa giác (polygon) kèm hai
  ràng buộc — *lớp phương tiện được phép* (ví dụ làn chỉ dành cho xe máy) và
  *hướng di chuyển hợp lệ*. Hệ thống xác định phương tiện thuộc làn nào theo tâm
  khung bao, rồi kiểm tra: nếu phương tiện sai lớp được phép, hoặc hướng di
  chuyển (vector dịch chuyển giữa các khung hình) lệch quá ngưỡng so với hướng
  hợp lệ của làn (đo bằng tích vô hướng – dot product), thì bị đánh dấu vi phạm.

Để chống cảnh báo nhiễu, mọi luật đều yêu cầu điều kiện vi phạm duy trì ổn định
qua một số khung hình liên tiếp tối thiểu trước khi xác nhận, đồng thời áp dụng
cơ chế thời gian nghỉ (cooldown) giữa hai lần cảnh báo liên tiếp cho cùng một vi
phạm trên cùng một phương tiện. Kết quả vi phạm được hiển thị trực tiếp trên giao
diện giám sát (mục 3.6).

## 3.5 Dự đoán hướng di chuyển và suy diễn hành trình

UC3 yêu cầu, từ quỹ đạo chuyển động của phương tiện và cấu trúc giao lộ trên bản
đồ, dự đoán hướng di chuyển tiếp theo của phương tiện và trực quan hoá trên bản
đồ. Bài toán được chia thành hai phần: dự đoán hướng rẽ tại ngã tư kế tiếp (mục
3.5.1) và suy diễn chuỗi đoạn đường tiếp theo trên mạng lưới đường thực tế (mục
3.5.2).

### 3.5.1 Dự đoán hướng rẽ tại ngã tư

Dự đoán hướng đi (đi thẳng/rẽ trái/rẽ phải/quay đầu) của một phương tiện dựa trên
một cửa sổ quan sát chuyển động gần nhất, về bản chất là một bài toán phân loại
chuỗi thời gian (time-series classification). Hai nhóm hướng tiếp cận thường
được dùng: (i) mạng học sâu tuần tự (LSTM/GRU, Transformer) xử lý trực tiếp chuỗi
toạ độ thô, có khả năng tự học biểu diễn thời gian phức tạp nhưng đòi hỏi lượng
dữ liệu huấn luyện lớn để tránh quá khớp; và (ii) các mô hình học máy cổ điển
trên một tập đặc trưng được thiết kế thủ công (engineered features) rút ra từ
chuỗi quan sát — như vận tốc trung bình, tổng biến động hướng đi, tỉ lệ giảm tốc
— rồi đưa vào một mô hình phân loại dạng bảng như Random Forest hay Gradient
Boosting [13].

Đề tài lựa chọn hướng (ii): **Gradient Boosting Classifier** (thư viện
`scikit-learn`), kết hợp một bước hiệu chỉnh xác suất đầu ra
(`CalibratedClassifierCV`, dựa trên nguyên lý Platt scaling [14]), trên một tập
33 đặc trưng thiết kế thủ công mô tả động học của phương tiện. Lý do chính là quy
mô dữ liệu gán nhãn thật hiện có (thu được từ việc tự động gán nhãn hướng rẽ cho
các track quan sát từ video CCTV thật) còn ở mức vài nghìn track — đủ để một mô
hình cây tăng cường học tốt trên một tập đặc trưng bám sát bản chất vật lý của
việc rẽ hướng, nhưng chưa đủ lớn để một mạng học sâu học biểu diễn từ đầu một
cách tin cậy mà không quá khớp. Bước hiệu chỉnh xác suất được thêm vào vì lý do
trực tiếp liên quan đến yêu cầu minh bạch (mục 2.4): một mô hình phân loại thông
thường có thể cho ra xác suất "quá tự tin" so với độ chính xác thật; Platt
scaling ánh xạ lại các điểm số thô thành xác suất phản ánh đúng hơn tần suất đúng
thực tế, đo bằng chỉ số sai số hiệu chỉnh kỳ vọng (Expected Calibration Error –
ECE) [15], cho phép hiển thị một mức tin cậy cho người vận hành mà không phóng
đại.

Song song, để có một ước lượng vị trí/hướng đi tức thời ngay cả khi chưa đủ lịch
sử cho mô hình phân loại trên, hệ thống dùng một bộ lọc Kalman [8] mô hình hoá
theo gia tốc không đổi (constant-acceleration), ước lượng vị trí/vận tốc tiếp
theo của phương tiện từ các quan sát rời rạc có nhiễu. So với một mạng dự đoán
quỹ đạo học sâu, bộ lọc Kalman có ưu điểm không phụ thuộc dữ liệu huấn luyện —
hoạt động ngay trên một camera/khu vực mới — và đủ chính xác cho dự đoán ngắn hạn
(vài giây), trong khi dự đoán dài hạn (qua nhiều ngã tư) được xử lý bằng cơ chế
suy diễn trên đồ thị đường ở mục 3.5.2.

### 3.5.2 Mạng lưới đường thực tế và chiếu toạ độ

Để suy diễn chuỗi đoạn đường tiếp theo mà phương tiện có thể đi qua, cần một cấu
trúc dữ liệu mô tả mạng lưới đường dưới dạng đồ thị (nút là giao lộ, cạnh là đoạn
đường). Có ba hướng xây dựng đồ thị này: (i) tự vẽ tay toạ độ giao lộ/đoạn đường
cho riêng khu vực camera — khả thi với khu vực nhỏ nhưng khó mở rộng; (ii) gọi
một dịch vụ bản đồ thương mại có API truy vấn cấu trúc đường — tiện nhưng phụ
thuộc hạn mức/chi phí và không phải lúc nào cũng có đủ thông tin kết nối giữa các
giao lộ; (iii) xây dựng từ dữ liệu bản đồ mở **OpenStreetMap** (OSM) [16], chuyển
đổi qua công cụ của bộ mô phỏng giao thông **SUMO** [17] sang một mạng lưới đường
có cấu trúc làn/giao lộ rõ ràng và biểu diễn theo chuẩn **OpenDRIVE** [18] — một
định dạng mô tả đường tiêu chuẩn trong đó thông tin kết nối giữa các giao lộ được
khai báo rõ ràng, dễ phân tích trực tiếp thành đồ thị nút–cạnh.

Đề tài lựa chọn hướng (iii). Dữ liệu OpenStreetMap cho khu vực đô thị Việt Nam
(bao gồm khu vực giám sát thực tế của đề tài) có sẵn, miễn phí và đủ chi tiết về
hình học đường; quy trình chuyển đổi qua SUMO/OpenDRIVE xử lý sẵn các vấn đề như
làn xe và hướng đi một chiều, cho ra một đồ thị giao lộ có thể duyệt được mà
không phụ thuộc dịch vụ bên thứ ba. Trên đồ thị này, hướng rẽ dự đoán ở ngã tư
đầu tiên (mục 3.5.1) được khớp với hướng hình học của các đoạn đường nối ra khỏi
giao lộ để chọn đoạn đường kế tiếp, lặp lại nhiều bước nhằm suy ra một hành trình
khả dĩ. Toạ độ trên đồ thị được quy đổi sang hệ toạ độ địa lý thực (vĩ độ/kinh
độ) để hiển thị trực tiếp lên nền bản đồ.

Một vấn đề liên quan là quy đổi toạ độ điểm ảnh (pixel) trên khung hình camera
sang toạ độ thực trên mặt đất, cần thiết để định vị phương tiện trên đồ thị giao
lộ. Đề tài dùng một mô hình chiếu hình ảnh theo mặt phẳng mặt đất (ground-plane
projection, dựa trên mô hình camera lỗ kim), với các tham số vị trí, góc nghiêng
và góc nhìn của camera được khớp bằng cách quan sát trực quan (đối chiếu mạng
lưới đường đã biết với hình ảnh thực tế từ camera) thay vì dùng thiết bị đo đạc
hay bảng caro hiệu chỉnh. Hướng hiệu chỉnh đầy đủ bằng thiết bị chuyên dụng cho
độ chính xác cao hơn nhưng không khả thi với phần lớn camera CCTV đang vận hành
(không thể tiếp cận vật lý để hiệu chỉnh lại); lựa chọn khớp trực quan vì vậy phù
hợp hơn với yêu cầu triển khai trên hạ tầng camera sẵn có (mục 2.4), đổi lại sai
số vị trí ở mức một vài mét — đủ để xác định đúng đoạn đường/giao lộ gần nhất
trên đồ thị.

## 3.6 Kiến trúc dịch vụ thời gian thực và trực quan hoá

UC2 (hiển thị vi phạm), UC3 (vẽ hành trình lên bản đồ) và UC4 (cấu hình hệ
thống) đều yêu cầu một lớp dịch vụ đẩy kết quả xử lý (luồng video đã vẽ chú
thích, danh sách vi phạm, hành trình dự đoán) tới giao diện người dùng theo thời
gian thực qua trình duyệt web, đồng thời nhận lại các cấu hình từ người vận hành.

Ở phía dịch vụ phía sau (backend), các lựa chọn phổ biến cho một dịch vụ Web
Python gồm Django REST Framework (đầy đủ nhưng thiên về xử lý đồng bộ, không tối
ưu cho nhiều kết nối streaming dài hạn), Flask kết hợp Flask-SocketIO, hoặc
FastAPI [19]. Đề tài lựa chọn **FastAPI** vì hỗ trợ lập trình bất đồng bộ
(async/await) một cách tự nhiên — phù hợp để vừa phục vụ API thông thường (REST:
quản lý nguồn video, vùng quan tâm, lịch sử vi phạm — phục vụ UC4), vừa duy trì
kết nối **WebSocket** [20] đẩy cảnh báo vi phạm và số liệu thống kê theo thời
gian thực, vừa truyền luồng hình ảnh camera dưới dạng **MJPEG** (Motion JPEG nối
tiếp qua HTTP, một kỹ thuật truyền hình ảnh đơn giản, hiển thị trực tiếp trên
trình duyệt mà không cần codec video phức tạp hay plugin). Pipeline AI (các mục
3.1–3.5) chạy trên một luồng nền và chia sẻ trạng thái với lớp dịch vụ; nhờ kiến
trúc bất đồng bộ, cả ba kênh dữ liệu trên cùng chạy trong một dịch vụ Python duy
nhất.

Ở phía giao diện người dùng (frontend), đề tài lựa chọn **React** [21] (kết hợp
công cụ đóng gói Vite và thư viện giao diện Tailwind CSS) thay cho Vue.js hay
Angular, vì mô hình quản lý trạng thái dựa trên hook phù hợp với việc đồng bộ
nhiều luồng dữ liệu thời gian thực độc lập (hình ảnh camera, danh sách vi phạm,
số liệu thống kê) cùng lúc trên một giao diện, cùng hệ sinh thái thư viện thành
phần phong phú giúp dựng nhanh lưới hiển thị nhiều camera và các panel cấu hình.
Để hiển thị hành trình dự đoán lên bản đồ (UC3), đề tài dùng **Leaflet** [22] —
một thư viện bản đồ JavaScript mã nguồn mở, nhẹ — kết hợp nền bản đồ (tile layer)
từ chính OpenStreetMap đã dùng để xây dựng đồ thị giao lộ ở mục 3.5.2, thay cho
các dịch vụ bản đồ thương mại. Lựa chọn này không yêu cầu khoá API hay chi phí sử
dụng, đồng thời giữ nhất quán một nguồn dữ liệu bản đồ duy nhất (OSM) cho cả phần
suy diễn hành trình và phần hiển thị, tránh sai lệch khi dùng hai nguồn bản đồ
khác nhau cho hai mục đích liên quan trực tiếp với nhau.

Về khả năng đáp ứng thời gian thực nhiều luồng camera (mục 2.4), pipeline áp dụng
hai kỹ thuật chính. Thứ nhất, **phát hiện theo lô (batch detection)**: các khung
hình từ nhiều camera được gom lại và đưa qua mô hình phát hiện trong một lượt suy
luận, tận dụng tối đa năng lực song song của GPU thay vì chạy tuần tự từng
camera. Thứ hai, **điều tiết tần suất các bước nặng**: các bước tốn tài nguyên
được chạy thưa hơn (không nhất thiết mỗi khung hình) và tách khỏi luồng giao
diện, giúp duy trì mức ~15–20 FPS trên cấu hình phần cứng thông thường khi xử lý
nhiều luồng video mô phỏng CCTV đồng thời.

---

Chương này đã trình bày các nền tảng lý thuyết và công nghệ được lựa chọn cho
từng thành phần của hệ thống, cùng các hướng tiếp cận thay thế đã được cân nhắc.
Có thể tổng kết lựa chọn theo bốn nguyên tắc xuyên suốt, bám sát các yêu cầu ở
Chương 2: ưu tiên mô hình nhẹ, suy luận nhanh để đáp ứng yêu cầu thời gian thực
trên nhiều camera (YOLO11, ByteTrack); ưu tiên các phương pháp ít phụ thuộc dữ
liệu huấn luyện lớn, phù hợp với lượng dữ liệu thật hiện có (Gradient Boosting có
hiệu chỉnh xác suất, bộ lọc Kalman, hệ luật suy diễn); ưu tiên tính minh bạch và
khả năng diễn giải của kết quả (hiệu chỉnh xác suất, hệ luật thay cho học máy
không giám sát); và ưu tiên dữ liệu/công cụ mở, áp dụng trực tiếp lên hạ tầng
camera CCTV thật (OpenStreetMap, SUMO, OpenDRIVE, FastAPI, React, Leaflet).
Chương 4 sẽ trình bày chi tiết quá trình thiết kế, triển khai và đánh giá hệ
thống dựa trên các nền tảng công nghệ đã giới thiệu.

---

## TÀI LIỆU THAM KHẢO (Chương 3)

[1] K. He, G. Gkioxari, P. Dollár, R. Girshick, "Mask R-CNN," in *Proc. IEEE
International Conference on Computer Vision (ICCV)*, 2017, pp. 2961–2969.

[2] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once:
Unified, Real-Time Object Detection," in *Proc. IEEE Conference on Computer
Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[3] G. Jocher, J. Qiu, "Ultralytics YOLO11." [Online]. Available:
https://docs.ultralytics.com/models/yolo11/ (truy cập 06/2026).

[4] W. Liu, D. Anguelov, D. Erhan, et al., "SSD: Single Shot MultiBox
Detector," in *Proc. European Conference on Computer Vision (ECCV)*, 2016,
pp. 21–37.

[5] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, S. Belongie,
"Feature Pyramid Networks for Object Detection," in *Proc. IEEE Conference on
Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 2117–2125.

[6] S. Liu, L. Qi, H. Qin, J. Shi, J. Jia, "Path Aggregation Network for
Instance Segmentation," in *Proc. IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)*, 2018, pp. 8759–8768.

[7] S. J. Pan, Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on
Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, 2010.

[8] R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems,"
*Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.

[9] Y. Zhang, P. Sun, Y. Jiang, et al., "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box," in *Proc. European Conference on Computer
Vision (ECCV)*, 2022, pp. 1–21.

[10] M. Broström, "BoxMOT: Pluggable SOTA Multi-Object Tracking Modules."
[Online]. Available: https://github.com/mikel-brostrom/boxmot (truy cập
06/2026).

[11] B. T. Morris, M. M. Trivedi, "A Survey of Vision-Based Trajectory Learning
and Analysis for Surveillance," *IEEE Transactions on Circuits and Systems for
Video Technology*, vol. 18, no. 8, pp. 1114–1127, 2008.

[12] F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE
International Conference on Data Mining (ICDM)*, 2008, pp. 413–422.

[13] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting
Machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[14] J. Platt, "Probabilistic Outputs for Support Vector Machines and
Comparisons to Regularized Likelihood Methods," *Advances in Large Margin
Classifiers*, vol. 10, no. 3, pp. 61–74, 1999.

[15] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On Calibration of Modern
Neural Networks," in *Proc. International Conference on Machine Learning
(ICML)*, 2017, pp. 1321–1330.

[16] OpenStreetMap contributors, "OpenStreetMap." [Online]. Available:
https://www.openstreetmap.org (truy cập 06/2026).

[17] P. A. Lopez, M. Behrisch, L. Bieker-Walz, et al., "Microscopic Traffic
Simulation using SUMO," in *Proc. IEEE International Conference on Intelligent
Transportation Systems (ITSC)*, 2018, pp. 2575–2582.

[18] ASAM e.V., "OpenDRIVE Format Specification," version 1.7, 2021.

[19] S. Ramírez, "FastAPI." [Online]. Available: https://fastapi.tiangolo.com/
(truy cập 06/2026).

[20] I. Fette, A. Melnikov, "The WebSocket Protocol," *RFC 6455, Internet
Engineering Task Force (IETF)*, 2011.

[21] Meta Platforms, Inc., "React – The Library for Web and Native User
Interfaces." [Online]. Available: https://react.dev/ (truy cập 06/2026).

[22] V. Agafonkin, "Leaflet – An Open-Source JavaScript Library for
Mobile-Friendly Interactive Maps." [Online]. Available: https://leafletjs.com/
(truy cập 06/2026).

[23] T.-Y. Lin, M. Maire, S. Belongie, et al., "Microsoft COCO: Common Objects
in Context," in *Proc. European Conference on Computer Vision (ECCV)*, 2014,
pp. 740–755.

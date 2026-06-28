# CHƯƠNG 3. NỀN TẢNG LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG

> Chương 2 đã xác định các yêu cầu chức năng của hệ thống — phát hiện và theo
> dõi đối tượng trong từng camera (UC1.1–UC1.2), phát hiện và đánh dấu sự cố
> (UC3), dự đoán hướng rẽ và suy diễn hành trình dài hạn trên bản đồ thực tế
> (UC4.1–UC4.3) — cùng các yêu cầu phi chức năng về hiệu năng thời gian thực,
> khả năng mở rộng nguồn camera (gồm năng lực duy trì định danh xuyên
> camera, mục 2.4), tính minh bạch của dự đoán và khả năng triển khai trực
> tiếp trên dữ liệu CCTV thật.
> Chương 3 trình bày các nền tảng lý thuyết và công nghệ được lựa chọn để
> hiện thực hoá từng yêu cầu trên, đồng thời phân tích các hướng tiếp cận
> thay thế đã được xem xét và lý do lựa chọn cuối cùng. Nội dung được trình
> bày theo đúng thứ tự luồng xử lý đã giới thiệu ở Chương 2: phát hiện đối
> tượng (3.1), theo dõi trong một camera (3.2), nhận diện lại xuyên camera
> (3.3), phát hiện sự cố (3.4), dự đoán hành vi phương tiện (3.5), xây dựng
> mạng lưới đường thực tế và suy diễn hành trình dài hạn (3.6), và kiến trúc
> dịch vụ thời gian thực/trực quan hoá (3.7).

---

## 3.1 Phát hiện đối tượng

Bài toán phát hiện đối tượng (object detection) yêu cầu đồng thời xác định vị
trí (qua khung bao — bounding box) và phân loại đối tượng xuất hiện trong
ảnh. Đây là nền tảng trực tiếp cho UC1.1 (Chương 2), vì mọi mô-đun phía sau
(theo dõi, nhận diện lại, dự đoán hành vi) đều chỉ hoạt động được trên đầu ra
của bước phát hiện này; đồng thời tốc độ xử lý ở bước này quyết định phần
lớn khả năng đáp ứng yêu cầu hiệu năng thời gian thực đã đặt ra ở mục 2.4.

Các hướng tiếp cận phổ biến cho bài toán này gồm: (i) kiến trúc hai giai
đoạn (two-stage) như Faster R-CNN/Mask R-CNN, trong đó một mạng đề xuất vùng
(region proposal) chạy trước một mạng phân loại/định vị tinh, cho độ chính
xác cao nhưng tốc độ suy luận chậm, không phù hợp xử lý nhiều luồng camera
đồng thời theo thời gian thực [1]; (ii) kiến trúc một giai đoạn (one-stage)
như SSD và EfficientDet, dự đoán trực tiếp khung bao và lớp trên một lượt
duyệt mạng, nhanh hơn rõ rệt nhưng thường đánh đổi độ chính xác trên các đối
tượng nhỏ — vấn đề đáng kể với xe máy quan sát từ xa qua camera CCTV; và
(iii) họ mô hình YOLO (You Only Look Once) [2], cũng thuộc nhóm một giai
đoạn nhưng liên tục được cải tiến qua các phiên bản để thu hẹp khoảng cách
độ chính xác với mô hình hai giai đoạn trong khi vẫn giữ tốc độ suy luận
cao.

Đề tài lựa chọn **YOLO11** (Ultralytics) [3], cụ thể là biến thể nhỏ
(*small*), cho mô-đun phát hiện. Kiến trúc YOLO11 gồm ba phần: một backbone
trích xuất đặc trưng đa tỉ lệ, một neck tổng hợp đặc trưng theo kiểu kim tự
tháp đặc trưng (feature pyramid) để vừa giữ ngữ nghĩa cấp cao vừa giữ chi
tiết không gian cấp thấp — quan trọng để phát hiện được xe máy nhỏ và xa lẫn
ô tô/xe buýt lớn trong cùng một khung hình — và một detection head theo
kiểu *anchor-free*, dự đoán trực tiếp toạ độ khung bao mà không cần các hộp
neo cố định được thiết kế trước. Lý do lựa chọn YOLO11 thay cho hai hướng
còn lại: (1) tốc độ suy luận đủ nhanh để xử lý nhiều camera trên phần cứng
thông thường, không yêu cầu máy chủ GPU chuyên dụng — đáp ứng trực tiếp yêu
cầu hiệu năng và khả năng triển khai trên dữ liệu thực tế (mục 2.4); (2) hệ
sinh thái Ultralytics cung cấp sẵn công cụ huấn luyện lại (fine-tuning) trên
bộ nhãn tuỳ chỉnh, cho phép huấn luyện lại mô hình theo đúng 5 lớp đối tượng
đặc trưng giao thông Việt Nam (người đi bộ, xe máy, ô tô, xe buýt, xe tải)
thay vì dùng nguyên bộ nhãn COCO gốc; (3) so với các phiên bản YOLO trước
(YOLOv5/v8), YOLO11 cải thiện độ chính xác trên vật thể nhỏ mà không tăng
đáng kể độ trễ suy luận — phù hợp đặc thù xe máy chiếm tỷ trọng lớn nhưng
kích thước nhỏ trong ảnh CCTV.

## 3.2 Theo dõi đa đối tượng trong một camera

Sau khi có danh sách đối tượng phát hiện ở mỗi khung hình, hệ thống cần liên
kết các phát hiện đó qua các khung hình liên tiếp để biết "đây vẫn là cùng
một đối tượng" — đây chính là yêu cầu của UC1.2. Bài toán theo dõi đa đối
tượng (Multi-Object Tracking — MOT) trong một camera thường theo mô hình
*tracking-by-detection*: dùng một bộ lọc chuyển động (thường là Kalman
Filter [4]) để dự đoán vị trí tiếp theo của các đối tượng đang theo dõi, rồi
ghép cặp các phát hiện mới với các dự đoán đó bằng thuật toán tối ưu hoá
ghép cặp hai phía (Hungarian matching).

Các hướng tiếp cận khác nhau ở chỗ ngoài chuyển động, có sử dụng thêm đặc
trưng hình ảnh (appearance) để ghép cặp hay không. SORT chỉ dùng độ chồng
lấp khung bao (IoU) và Kalman Filter, đơn giản và nhanh nhưng dễ mất dấu khi
đối tượng bị che khuất. DeepSORT và StrongSORT bổ sung một mạng trích đặc
trưng hình ảnh riêng cho từng đối tượng ở *mỗi khung hình* để ghép cặp chính
xác hơn khi chuyển động không đủ tin cậy, nhưng phải trả giá bằng chi phí
tính toán tăng thêm đáng kể vì phải chạy một mạng trích đặc trưng cho từng
đối tượng liên tục — chi phí này có thể chấp nhận được cho một bài toán đơn
camera, nhưng sẽ dư thừa cho hệ thống của đề tài vì một mạng trích đặc trưng
chuyên biệt khác (OSNet, mục 3.3) đã được dùng riêng cho bước theo dõi xuyên
camera. ByteTrack [5] đại diện cho một cải tiến khác: thay vì chỉ giữ các
phát hiện có độ tin cậy cao, ByteTrack giữ và ghép cặp *toàn bộ* khung bao
phát hiện được, kể cả các khung bao có độ tin cậy thấp (chiến lược BYTE),
dựa trên lập luận rằng một phát hiện điểm tin cậy thấp do bị che khuất một
phần vẫn là tín hiệu hữu ích để duy trì track, chỉ cần không dùng nó để khởi
tạo track mới.

Đề tài lựa chọn **ByteTrack**, triển khai qua thư viện `boxmot` [6]. Việc
giữ lại các phát hiện độ tin cậy thấp đặc biệt phù hợp với giao thông Việt
Nam, nơi xe máy di chuyển sát nhau và thường xuyên che khuất lẫn nhau một
phần trong khung hình CCTV, khiến độ tin cậy phát hiện dao động liên tục —
nếu loại bỏ các phát hiện độ tin cậy thấp như SORT, track của xe máy bị che
khuất tạm thời sẽ dễ bị ngắt quãng và tạo ra ID mới một cách không cần
thiết. Đồng thời, vì ByteTrack chỉ dựa vào chuyển động (không cần mạng trích
đặc trưng hình ảnh riêng), chi phí tính toán ở bước theo dõi trong một
camera được giữ tối thiểu, dành tài nguyên tính toán cho bước nhận diện lại
xuyên camera ở mục 3.3 — vốn mới là nơi thực sự cần đặc trưng hình ảnh.

## 3.3 Nhận diện lại xuyên camera (Re-Identification)

Năng lực hạ tầng duy trì định danh xuyên camera (mục 2.4) yêu cầu hệ thống
nhận ra cùng một đối tượng khi nó xuất hiện ở một camera khác — bài toán này
được gọi là nhận diện lại (Re-Identification — ReID). Khác với theo dõi trong một camera (mục 3.2), ReID không có thông tin
chuyển động liên tục giữa hai camera (đối tượng "biến mất" khỏi camera này
một khoảng thời gian trước khi "xuất hiện" ở camera kia), nên phải dựa hoàn
toàn vào đặc trưng hình ảnh: trích một vector đặc trưng (embedding) đại diện
cho "diện mạo" của đối tượng, sao cho hai lần quan sát cùng một đối tượng dù
ở góc camera khác nhau vẫn cho ra hai vector gần nhau, còn hai đối tượng
khác nhau cho ra hai vector cách xa nhau — sau đó so khớp bằng độ tương đồng
cô-sin (cosine similarity).

Các hướng tiếp cận trích đặc trưng khác nhau ở mức độ chuyên biệt hoá cho
bài toán ReID. Dùng trực tiếp một mạng phân loại ảnh tổng quát (ví dụ
ResNet-50 huấn luyện trên ImageNet) làm bộ trích đặc trưng là cách đơn giản
nhất nhưng cho chất lượng phân biệt thấp vì mạng không được tối ưu cho việc
phân biệt các cá thể cùng lớp (cùng là "xe máy" nhưng là hai xe khác nhau).
Các kiến trúc theo hướng học đặc trưng theo từng phần cơ thể/phương tiện
(part-based, ví dụ PCB) hoặc kiến trúc Transformer cho ReID (TransReID) cho
độ chính xác cao hơn nhưng đòi hỏi nhiều tham số và dữ liệu huấn luyện lớn
hơn, gây khó khăn khi cần suy luận thời gian thực trên nhiều đối tượng ở
nhiều camera đồng thời.

Đề tài lựa chọn **OSNet** (Omni-Scale Network) [7], một kiến trúc CNN được
thiết kế riêng cho bài toán ReID với cơ chế tổng hợp đặc trưng đa tỉ lệ qua
các cổng tổng hợp thống nhất (unified aggregation gates), cho phép mạng học
được cả đặc trưng cục bộ chi tiết (màu sắc, hoa văn) và đặc trưng toàn cục
(hình dạng tổng thể) trong cùng một nhánh mạng nhẹ (~2,2 triệu tham số). Lý
do lựa chọn OSNet: (1) số tham số nhỏ giúp trích đặc trưng cho nhiều đối
tượng từ nhiều camera đồng thời mà không vượt quá khả năng phần cứng thông
thường, trực tiếp đáp ứng yêu cầu khả năng mở rộng số lượng camera (mục
2.4); (2) đã có sẵn trọng số huấn luyện trước (pretrained) trên bộ dữ liệu
Market-1501 [8] cho người đi bộ, đồng thời có thể huấn luyện lại (fine-tune)
trên VeRi-776 [9] — bộ dữ liệu ReID chuyên cho phương tiện cơ giới (ô tô,
xe máy) — thông qua thư viện `torchreid` [10], phù hợp với yêu cầu hạ tầng
Re-ID cần phân biệt nhiều loại đối tượng khác nhau (người đi bộ và nhiều
loại phương tiện) chứ không chỉ một loại duy nhất. Để giảm tỷ lệ ghép sai khi hai đối tượng có
diện mạo tương tự (ví dụ hai xe máy cùng màu, cùng kiểu), việc so khớp đặc
trưng còn được kết hợp với một bộ lọc thời gian–không gian: chỉ chấp nhận
một cặp match nếu khoảng thời gian giữa hai lần quan sát phù hợp với thời
gian di chuyển thực tế tối thiểu/tối đa giữa hai camera tương ứng (được khai
báo trước theo cấu trúc bố trí camera) — đây là một kỹ thuật ràng buộc theo
ngữ cảnh không gian–thời gian phổ biến trong các hệ thống theo dõi đa camera
quy mô lớn [11].

## 3.4 Phát hiện sự cố giao thông

UC3 yêu cầu hệ thống tự động phát hiện các tình huống bất thường (dừng đột
ngột, đi ngược chiều, vượt đèn đỏ, phương tiện áp sát nhau, người đi bộ vào
vùng nguy hiểm...) từ dữ liệu track output của các bước trên. Có hai hướng
tiếp cận chính cho bài toán phát hiện bất thường trong giám sát giao thông
[12]. Hướng thứ nhất là học máy không giám sát (unsupervised anomaly
detection): huấn luyện một mô hình (ví dụ Isolation Forest [13] hoặc bộ tự
mã hoá — autoencoder) để tự học "thế nào là hành vi bình thường" từ một
lượng lớn dữ liệu track quan sát được, rồi đánh dấu bất thường là những mẫu
sai khác nhiều so với phân phối đã học. Hướng thứ hai là hệ luật suy diễn
(rule-based reasoning): định nghĩa trực tiếp các ngưỡng/điều kiện cụ thể
(vận tốc, gia tốc, vị trí tương đối với vùng cấm...) ứng với từng loại sự cố
cần phát hiện.

Đề tài lựa chọn **hệ luật suy diễn** cho UC3.1. Lý do chính nằm ở yêu cầu
minh bạch của dự đoán (mục 2.4): với hướng học máy không giám sát, hệ thống
chỉ có thể báo "hành vi này khác thường" mà khó giải thích rõ vì sao, gây
khó khăn khi người vận hành cần hiểu căn cứ của một cảnh báo trước khi quyết
định hành động; trong khi với hệ luật, mỗi cảnh báo gắn liền với một điều
kiện cụ thể, có thể diễn giải trực tiếp (ví dụ "vận tốc giảm từ X xuống gần 0
trong dưới 1 giây" cho luật dừng đột ngột) và dễ điều chỉnh ngưỡng theo từng
camera/tuyến đường cụ thể mà không cần huấn luyện lại. Ngoài ra, hướng học
máy không giám sát đòi hỏi một lượng dữ liệu quan sát "bình thường" đủ lớn
và đặc thù cho từng địa điểm camera để mô hình học đúng phân phối nền —
trong khi sự cố giao thông thật vốn hiếm và khó thu thập đủ để kiểm chứng mô
hình học máy một cách tin cậy. Cơ chế thời gian nghỉ (cooldown) giữa hai lần
cảnh báo liên tiếp cho cùng một loại sự cố trên cùng một đối tượng được áp
dụng thêm để đáp ứng yêu cầu chống nhiễu cảnh báo. Hướng học máy không giám
sát vẫn được ghi nhận là một hướng nâng cấp khả thi trong tương lai khi đã
tích lũy đủ dữ liệu vận hành thực tế (trình bày lại ở Chương 6).

## 3.5 Dự đoán hành vi phương tiện

UC4.1 yêu cầu dự đoán hướng đi (đi thẳng/rẽ trái/rẽ phải/quay đầu) của một
phương tiện tại ngã tư sắp tới, dựa trên một cửa sổ quan sát chuyển động gần
nhất. Đây bản chất là một bài toán phân loại chuỗi thời gian (time-series
classification). Hai nhóm hướng tiếp cận thường được dùng: (i) mạng học sâu
hồi quy/tuần tự xử lý trực tiếp chuỗi toạ độ thô, như LSTM/GRU hoặc các kiến
trúc Transformer cho dự báo chuyển động, có khả năng tự học biểu diễn thời
gian phức tạp nhưng đòi hỏi lượng dữ liệu huấn luyện lớn để tránh quá khớp;
và (ii) các mô hình học máy cổ điển trên một tập đặc trưng được thiết kế thủ
công (engineered features) rút ra từ chuỗi quan sát — ví dụ vận tốc trung
bình, tổng biến động hướng đi, tỉ lệ giảm tốc trong cửa sổ quan sát — rồi
đưa vào một mô hình phân loại dạng bảng như Random Forest hay Gradient
Boosting [14].

Đề tài lựa chọn hướng (ii): **Gradient Boosting Classifier**, kết hợp một
bước hiệu chỉnh xác suất đầu ra (`CalibratedClassifierCV`, dựa trên nguyên
lý Platt scaling [15]). Lý do chính là quy mô dữ liệu gán nhãn thật hiện có
(thu được từ việc tự động gán nhãn hướng rẽ cho các track quan sát từ video
CCTV thật) còn ở mức vài nghìn track — đủ để một mô hình cây tăng cường học
tốt trên một tập đặc trưng đã được thiết kế có chủ đích bám sát bản chất vật
lý của việc rẽ hướng (biến động hướng đi, giảm tốc trước khi rẽ...), nhưng
chưa đủ lớn để một mạng hồi quy sâu học biểu diễn từ đầu một cách tin cậy mà
không quá khớp. Bước hiệu chỉnh xác suất (calibration) được thêm vào vì lý
do trực tiếp liên quan đến yêu cầu minh bạch dự đoán ở mục 2.4: một mô hình
phân loại thông thường có thể cho ra xác suất "quá tự tin" so với độ chính
xác thật của nó; Platt scaling ánh xạ lại các điểm số thô của mô hình thành
xác suất phản ánh đúng hơn tần suất đúng thực tế, đo bằng chỉ số sai số hiệu
chỉnh kỳ vọng (Expected Calibration Error — ECE) [16] — cho phép hiển thị
một mức tin cậy cho người vận hành mà không phóng đại.

Song song, để có một ước lượng vị trí/hướng đi tức thời (đầu vào cho UC4.1
và UC4.2) ngay cả khi chưa đủ lịch sử cho mô hình phân loại ở trên, hệ thống
dùng một bộ lọc Kalman (Kalman Filter) [4] mô hình hoá theo gia tốc không
đổi (constant-acceleration), ước lượng vị trí/vận tốc tiếp theo của đối
tượng từ các quan sát rời rạc có nhiễu. So với các phương án thay thế (mạng
dự đoán quỹ đạo học sâu đa giả thuyết kiểu Social-LSTM hoặc Transformer dự
báo chuyển động), Kalman Filter có ưu điểm không phụ thuộc dữ liệu huấn
luyện — hoạt động ngay trên một camera/khu vực mới mà không cần thu thập và
gán nhãn trước — và đủ chính xác cho horizon ngắn (vài giây), trong khi bài
toán dự đoán cho horizon dài (nhiều ngã tư) được xử lý bằng một cơ chế khác,
trình bày ở mục 3.6, để tránh việc một mô hình học sâu phải gánh đồng thời
cả dự đoán ngắn hạn chính xác và dự đoán dài hạn vốn cần dữ liệu lớn hơn
nhiều mới đáng tin cậy.

## 3.6 Xây dựng mạng lưới đường thực tế và suy diễn hành trình dài hạn

UC4.2 yêu cầu, từ hướng đi dự đoán ở ngã tư đầu tiên, tiếp tục suy diễn ra
chuỗi các ngã tư/đoạn đường tiếp theo mà đối tượng có thể đi qua, dựa trên
cấu trúc đường thực tế của khu vực giám sát — bài toán này cần một cấu trúc
dữ liệu mô tả mạng lưới đường dưới dạng đồ thị (nút là giao lộ, cạnh là đoạn
đường) để có thể duyệt tiếp.

Có ba hướng để xây dựng đồ thị này: (i) tự vẽ tay (manual annotation) toạ độ
các giao lộ/đoạn đường cho riêng khu vực camera đang giám sát — khả thi cho
một khu vực nhỏ nhưng không thể mở rộng nhanh sang khu vực mới; (ii) gọi một
dịch vụ bản đồ thương mại có API truy vấn cấu trúc đường (ví dụ Google Roads
API) — tiện lợi nhưng phụ thuộc vào hạn mức truy vấn/chi phí và định dạng dữ
liệu trả về không phải lúc nào cũng có đủ thông tin kết nối giữa các giao lộ
cần cho việc duyệt đồ thị; (iii) xây dựng từ dữ liệu bản đồ mở
**OpenStreetMap** (OSM) [17], chuyển đổi qua công cụ `netconvert` của bộ mô
phỏng giao thông **SUMO** [18] sang một mạng lưới đường có cấu trúc làn/giao
lộ rõ ràng, rồi biểu diễn theo chuẩn **OpenDRIVE** [19] — một định dạng mô
tả đường tiêu chuẩn trong đó thông tin kết nối giữa các giao lộ (junction
connectivity) được khai báo rõ ràng, dễ phân tích trực tiếp thành đồ thị nút
(giao lộ) – cạnh (đoạn đường) phục vụ thuật toán duyệt đồ thị.

Đề tài lựa chọn hướng (iii). Lý do: dữ liệu OpenStreetMap cho khu vực đô thị
Việt Nam (bao gồm khu vực giám sát thực tế của đề tài) có sẵn, miễn phí và
đủ chi tiết về hình học đường; `netconvert` là công cụ đã được dùng rộng
rãi trong nghiên cứu mô phỏng/phân tích giao thông để chuyển đổi dữ liệu OSM
thô thành một mạng lưới có cấu trúc, xử lý sẵn các vấn đề như làn xe, hướng
đi một chiều; còn OpenDRIVE cung cấp một lược đồ dữ liệu đã chuẩn hoá, có
thể phân tích bằng công cụ tự viết để trích ra đúng đồ thị giao lộ cần cho
UC4.2 mà không phụ thuộc dịch vụ bên thứ ba có thể giới hạn truy vấn hoặc
thay đổi điều khoản sử dụng. Toạ độ trên đồ thị này được quy đổi hai chiều
sang hệ toạ độ địa lý thực (vĩ độ/kinh độ) thông qua phép chiếu bản đồ UTM
tiêu chuẩn, để có thể hiển thị trực tiếp lên nền bản đồ thực ở UC4.3.

Một vấn đề liên quan là quy đổi toạ độ điểm ảnh (pixel) trên khung hình
camera sang toạ độ thực trên mặt đất, cần thiết để biết vị trí hiện tại của
đối tượng trên đồ thị giao lộ nói trên. Đề tài dùng một mô hình chiếu hình
ảnh đơn giản theo mặt phẳng mặt đất (ground-plane projection, dựa trên mô
hình camera lỗ kim — *pinhole camera*), với các tham số vị trí, góc nghiêng
và góc nhìn (FOV) của camera được khớp bằng cách quan sát trực quan (đối
chiếu mạng lưới đường đã biết với hình ảnh thực tế từ camera) thay vì hiệu
chỉnh bằng thiết bị đo đạc chuyên dụng hoặc bảng caro hiệu chỉnh camera
(camera calibration checkerboard). Hướng hiệu chỉnh đầy đủ bằng thiết bị
chuyên dụng cho độ chính xác cao hơn nhưng không khả thi với phần lớn camera
CCTV đang vận hành (không thể tiếp cận vật lý để hiệu chỉnh lại); lựa chọn
khớp trực quan vì vậy phù hợp hơn với yêu cầu khả năng triển khai trực tiếp
trên hạ tầng camera thực tế đã có sẵn (mục 2.4), đổi lại sai số vị trí ước
lượng được ở mức một vài mét — đủ dùng để xác định đúng đoạn đường/giao lộ
gần nhất trên đồ thị, dù chưa đạt độ chính xác phục vụ mục đích đo đạc kỹ
thuật.

## 3.7 Kiến trúc dịch vụ thời gian thực và trực quan hoá

Cuối cùng, UC5 (giám sát qua dashboard) và UC4.3 (vẽ hành trình lên bản đồ)
yêu cầu một lớp dịch vụ đẩy được kết quả xử lý (luồng video đã vẽ chú thích,
danh sách cảnh báo, hành trình dự đoán) tới giao diện người dùng theo thời
gian thực qua trình duyệt web, đúng như yêu cầu phi chức năng về giao diện
vận hành thời gian thực (mục 2.4).

Ở phía dịch vụ phía sau (backend), các lựa chọn phổ biến cho một dịch vụ Web
Python gồm Django REST Framework (đầy đủ tính năng nhưng thiên về mô hình xử
lý đồng bộ, không tối ưu cho việc duy trì nhiều kết nối streaming dài hạn),
Flask kết hợp Flask-SocketIO, hoặc FastAPI [20]. Đề tài lựa chọn **FastAPI**
vì hỗ trợ lập trình bất đồng bộ (async/await) một cách tự nhiên — phù hợp để
vừa phục vụ API thông thường (REST, quản lý camera/vùng quan tâm/lịch sử cảnh
báo), vừa duy trì kết nối **WebSocket** [21] đẩy cảnh báo và số liệu thống
kê theo thời gian thực, vừa truyền luồng hình ảnh camera dưới dạng
**MJPEG** (Motion JPEG nối tiếp qua HTTP, một kỹ thuật truyền hình ảnh đơn
giản không cần codec video phức tạp, phù hợp hiển thị trực tiếp trên trình
duyệt mà không cần plugin) — cả ba đều chạy trong cùng một dịch vụ Python,
không cần tách riêng ngôn ngữ/dịch vụ với pipeline AI ở các mục 3.1–3.6.

Ở phía giao diện người dùng (frontend), đề tài lựa chọn **React** [22] thay
cho các framework khác như Vue.js hoặc Angular, vì mô hình quản lý trạng
thái dựa trên hook phù hợp với việc đồng bộ nhiều luồng dữ liệu thời gian
thực độc lập (hình ảnh camera, danh sách cảnh báo, số liệu thống kê) cùng
lúc trên một giao diện, cùng hệ sinh thái thư viện thành phần (component)
phong phú giúp xây dựng nhanh các phần hiển thị dạng lưới camera, panel cảnh
báo. Để hiển thị hành trình dự đoán lên bản đồ (UC4.3), đề tài dùng
**Leaflet** [23] — một thư viện bản đồ JavaScript mã nguồn mở, nhẹ — kết hợp
nền bản đồ (tile layer) từ chính OpenStreetMap đã dùng để xây dựng đồ thị
giao lộ ở mục 3.6, thay cho các dịch vụ bản đồ thương mại như Google Maps
JavaScript API hay Mapbox GL JS. Lựa chọn này vừa không yêu cầu khoá API hay
chi phí sử dụng, vừa giữ nhất quán một nguồn dữ liệu bản đồ duy nhất (OSM)
cho cả phần suy diễn hành trình (mục 3.6) và phần hiển thị trực quan, tránh
sai lệch có thể xảy ra nếu dùng hai nguồn bản đồ khác nhau cho hai mục đích
liên quan trực tiếp đến nhau.

---

Chương này đã trình bày các nền tảng lý thuyết và công nghệ được lựa chọn
cho từng thành phần của hệ thống, cùng các hướng tiếp cận thay thế đã được
cân nhắc. Có thể tổng kết lựa chọn theo bốn nguyên tắc xuyên suốt, bám sát
các yêu cầu đã đặt ra ở Chương 2: ưu tiên mô hình nhẹ, suy luận nhanh để đáp
ứng yêu cầu thời gian thực trên nhiều camera (YOLO11, ByteTrack, OSNet);
ưu tiên các phương pháp không hoặc ít phụ thuộc dữ liệu huấn luyện lớn, phù
hợp với lượng dữ liệu thật hiện có (Gradient Boosting có hiệu chỉnh xác
suất, Kalman Filter, hệ luật suy diễn); ưu tiên tính minh bạch và khả năng
diễn giải của kết quả đầu ra (hiệu chỉnh xác suất, hệ luật thay cho học máy
không giám sát); và ưu tiên dữ liệu/công cụ mở, có thể áp dụng trực tiếp lên
hạ tầng camera CCTV thật mà không cần thiết bị hay dịch vụ chuyên dụng
(OpenStreetMap, SUMO, OpenDRIVE, Leaflet). Chương 4 sẽ trình bày chi tiết quá
trình thiết kế, triển khai và đánh giá hệ thống dựa trên các nền tảng công
nghệ đã giới thiệu ở chương này.

---

## TÀI LIỆU THAM KHẢO (Chương 3)

[1] K. He, G. Gkioxari, P. Dollár, R. Girshick, "Mask R-CNN," in *Proc. IEEE
International Conference on Computer Vision (ICCV)*, 2017, pp. 2961–2969.

[2] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once:
Unified, Real-Time Object Detection," in *Proc. IEEE Conference on Computer
Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[3] G. Jocher, J. Qiu, "Ultralytics YOLO11." [Online]. Available:
https://docs.ultralytics.com/models/yolo11/ (truy cập 06/2026).

[4] R. E. Kalman, "A New Approach to Linear Filtering and Prediction
Problems," *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.

[5] Y. Zhang, P. Sun, Y. Jiang, et al., "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box," in *Proc. European Conference on Computer
Vision (ECCV)*, 2022, pp. 1–21.

[6] M. Broström, "BoxMOT: Pluggable SOTA Multi-Object Tracking Modules for
Segmentation, Object Detection and Pose Estimation Models." [Online].
Available: https://github.com/mikel-brostrom/boxmot (truy cập 06/2026).

[7] K. Zhou, Y. Yang, A. Cavallaro, T. Xiang, "Omni-Scale Feature Learning
for Person Re-Identification," in *Proc. IEEE International Conference on
Computer Vision (ICCV)*, 2019, pp. 3702–3712.

[8] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian, "Scalable Person
Re-identification: A Benchmark," in *Proc. IEEE International Conference on
Computer Vision (ICCV)*, 2015, pp. 1116–1124.

[9] X. Liu, W. Liu, H. Ma, H. Fu, "Large-Scale Vehicle Re-Identification in
Urban Surveillance Videos," in *Proc. IEEE International Conference on
Multimedia and Expo (ICME)*, 2016, pp. 1–6.

[10] K. Zhou, T. Xiang, "Torchreid: A Library for Deep Learning Person
Re-Identification in Pytorch," *arXiv preprint arXiv:1910.10093*, 2019.

[11] Z. Tang, M. Naphade, M.-Y. Liu, et al., "CityFlow: A City-Scale
Benchmark for Multi-Target Multi-Camera Vehicle Tracking and
Re-Identification," in *Proc. IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)*, 2019, pp. 8797–8806.

[12] B. T. Morris, M. M. Trivedi, "A Survey of Vision-Based Trajectory
Learning and Analysis for Surveillance," *IEEE Transactions on Circuits and
Systems for Video Technology*, vol. 18, no. 8, pp. 1114–1127, 2008.

[13] F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE
International Conference on Data Mining (ICDM)*, 2008, pp. 413–422.

[14] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting
Machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[15] J. Platt, "Probabilistic Outputs for Support Vector Machines and
Comparisons to Regularized Likelihood Methods," *Advances in Large Margin
Classifiers*, vol. 10, no. 3, pp. 61–74, 1999.

[16] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On Calibration of Modern
Neural Networks," in *Proc. International Conference on Machine Learning
(ICML)*, 2017, pp. 1321–1330.

[17] OpenStreetMap contributors, "OpenStreetMap." [Online]. Available:
https://www.openstreetmap.org (truy cập 06/2026).

[18] P. A. Lopez, M. Behrisch, L. Bieker-Walz, et al., "Microscopic Traffic
Simulation using SUMO," in *Proc. IEEE International Conference on
Intelligent Transportation Systems (ITSC)*, 2018, pp. 2575–2582.

[19] ASAM e.V., "OpenDRIVE Format Specification," version 1.7, 2021.

[20] S. Ramírez, "FastAPI." [Online]. Available:
https://fastapi.tiangolo.com/ (truy cập 06/2026).

[21] I. Fette, A. Melnikov, "The WebSocket Protocol," *RFC 6455, Internet
Engineering Task Force (IETF)*, 2011.

[22] Meta Platforms, Inc., "React – The Library for Web and Native User
Interfaces." [Online]. Available: https://react.dev/ (truy cập 06/2026).

[23] V. Agafonkin, "Leaflet – An Open-Source JavaScript Library for
Mobile-Friendly Interactive Maps." [Online]. Available:
https://leafletjs.com/ (truy cập 06/2026).

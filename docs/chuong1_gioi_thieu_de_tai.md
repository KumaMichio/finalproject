# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

> Chương này trình bày bốn nội dung mở đầu của đồ án. Phần 1.1 đặt vấn đề về
> hiện trạng giám sát giao thông bằng camera tại Việt Nam; phần 1.2 trình bày
> mục tiêu và phạm vi đề tài trên cơ sở đối chiếu với các hướng giải quyết
> liên quan hiện có; phần 1.3 nêu định hướng giải pháp và đóng góp chính của
> đồ án; phần 1.4 giới thiệu bố cục của các chương còn lại.

---

## 1.1 Đặt vấn đề

Tại các đô thị lớn của Việt Nam, đặc biệt là Hà Nội và Thành phố Hồ Chí
Minh, số lượng camera giám sát lắp đặt tại các giao lộ và tuyến đường chính
đã tăng nhanh trong những năm gần đây, đi cùng với tốc độ đô thị hoá và mật
độ phương tiện giao thông — trong đó xe máy chiếm tỷ trọng áp đảo — ngày
càng cao. Phần lớn hệ thống camera này vẫn hoạt động theo mô hình ghi hình
và truyền hình ảnh thuần túy: hình ảnh từ nhiều camera được đổ về một phòng
điều khiển trung tâm và do con người trực tiếp quan sát qua một lưới màn
hình.

Mô hình giám sát thụ động này bộc lộ ba hạn chế gắn liền với cách vận hành
của nó. Thứ nhất, một người vận hành không thể quan sát liên tục đồng thời
nhiều màn hình, nên phần lớn vi phạm hoặc sự cố giao thông chỉ được phát
hiện sau khi đã xảy ra, qua xem lại đoạn ghi hình, thay vì được cảnh báo
ngay tại thời điểm diễn ra. Thứ hai, mỗi camera hoạt động độc lập và không
lưu giữ được "trí nhớ" về một phương tiện cụ thể: khi phương tiện đó di
chuyển ra khỏi khung hình của camera này và xuất hiện ở một camera khác,
hệ thống không có cách nào tự động khẳng định đó là cùng một đối tượng, và
việc xác nhận lại lai lịch của đối tượng — kể cả khi đã ghi nhận được vi
phạm hoặc đánh dấu là đối tượng nghi vấn — phải do người vận hành tự tay rà
soát qua nhiều đoạn video riêng lẻ. Thứ ba, hệ thống chỉ phản ứng khi sự
việc đã xảy ra, không có khả năng dự đoán phương tiện sẽ tiếp tục di chuyển
theo hướng nào trong những ngã tư tiếp theo, trong khi đây là thông tin cần
thiết để lực lượng chức năng quyết định vị trí chốt chặn phù hợp sau khi
một phương tiện vi phạm hoặc gây sự cố rồi rời khỏi hiện trường.

Nếu các hạn chế trên được khắc phục, lực lượng cảnh sát giao thông và các
cơ quan quản lý đô thị sẽ có thêm một công cụ hỗ trợ phát hiện vi phạm liên
tục theo thời gian thực, truy vết một đối tượng cụ thể xuyên suốt nhiều
camera mà không phụ thuộc hoàn toàn vào việc rà soát thủ công, và có cơ sở
định lượng để dự đoán hướng di chuyển tiếp theo của đối tượng nhằm hỗ trợ ra
quyết định xử lý kịp thời. Cách tiếp cận xử lý ảnh và suy diễn hành vi từ
luồng camera theo thời gian thực không chỉ giới hạn ở bài toán giao thông,
mà còn có thể áp dụng cho các bài toán giám sát an ninh đô thị khác có cùng
đặc điểm dữ liệu đầu vào là nhiều luồng camera cố định quan sát một khu vực
chung.

## 1.2 Mục tiêu và phạm vi đề tài

Hướng nghiên cứu theo dõi đa đối tượng trong một camera đã khá trưởng
thành, với các thuật toán theo mô hình tracking-by-detection như DeepSORT,
StrongSORT và ByteTrack, nhưng các thuật toán này chỉ giải quyết bài toán
trong phạm vi một camera và không có cơ chế nối lại định danh khi đối tượng
rời khỏi khung hình. Hướng nghiên cứu theo dõi đa camera, đa mục tiêu, với
đại diện tiêu biểu là CityFlow, giải quyết được vấn đề "trí nhớ xuyên
camera" bằng một mô-đun nhận diện lại dựa trên đặc trưng hình ảnh, nhưng các
bộ dữ liệu và mô hình công khai trong hướng này được xây dựng trên dữ liệu
giao thông phương Tây, với cơ cấu phương tiện gần như không có xe máy, đồng
thời chỉ tập trung vào bài toán định danh mà không đi tiếp đến dự đoán
hướng đi tương lai trên một bản đồ đường thực tế. Gần với đúng bối cảnh
giao thông Việt Nam hơn, Chau et al. [1] giải quyết bài toán dự đoán quỹ
đạo phương tiện trực tiếp trên video CCTV tại Hà Nội/Thành phố Hồ Chí Minh,
nhưng toàn bộ phương pháp chỉ hoạt động trong phạm vi một camera duy nhất,
dự đoán toạ độ trong một khoảng thời gian ngắn, không có bước nhận diện lại
xuyên camera và không suy diễn hành trình qua nhiều ngã tư liên tiếp. Ở
nhóm sản phẩm thương mại, các nền tảng giám sát giao thông tại Việt Nam
hiện nay — hệ thống camera phạt nguội, hệ thống đèn tín hiệu "thông minh" —
chủ yếu phục vụ đếm lưu lượng phương tiện và phát hiện vi phạm để xử phạt
trên từng camera/giao lộ riêng lẻ; các hệ thống này có đọc và lưu biển số
xe vi phạm nhưng chỉ để phục vụ xử phạt hành chính ngay tại nơi ghi nhận,
không công khai khả năng theo dõi liên tục một phương tiện cụ thể qua nhiều
giao lộ và không có chức năng dự đoán hành trình tiếp theo của phương tiện.

Tổng hợp lại, chưa có giải pháp nào — học thuật hay thương mại — kết hợp
đồng thời ba năng lực trên cùng một hệ thống vận hành được trên dữ liệu
giao thông thật của Việt Nam: phát hiện vi phạm giao thông cụ thể gắn liền
với một đối tượng có thể truy vết, đánh dấu đối tượng nghi vấn kèm lưu lại
biển số xe khi điều kiện hình ảnh cho phép, và dự đoán hướng đi dài hạn của
phương tiện qua nhiều ngã tư liên tiếp trên một bản đồ đường thực tế. Đề
tài này hướng tới xây dựng một hệ thống giám sát giao thông bằng camera AI
nhằm thu hẹp khoảng trống trên, với phạm vi cụ thể là một hệ thống gồm 6
camera giám sát chạy đồng thời, vận hành ổn định trên video CCTV thật (hoặc
video giả lập CCTV dùng cho kiểm thử) của một tuyến đường có nhiều giao lộ
tại Hà Nội. Phần mềm cần có ba chức năng chính: phát hiện và theo dõi
phương tiện theo thời gian thực trong từng camera kèm phán đoán vi phạm
giao thông (vượt đèn đỏ/vượt vạch dừng đèn đỏ, đi sai làn); đánh dấu thủ
công một đối tượng nghi vấn và lưu lại biển số xe của đối tượng đó khi đọc
được; và dự đoán hướng đi dài hạn của phương tiện qua nhiều ngã tư liên
tiếp, hiển thị trực quan trên bản đồ đường thực tế.

## 1.3 Định hướng giải pháp

Đề tài giải quyết bài toán trên theo hướng kết hợp bốn nhóm kỹ thuật: phát
hiện đối tượng bằng học sâu kết hợp theo dõi theo phát hiện (tracking-by-
detection) để xử lý từng luồng camera theo thời gian thực; suy diễn theo
luật (rule-based) trên dữ liệu chuyển động của từng đối tượng để phát hiện
vi phạm và sự cố giao thông; nhận diện lại bằng đặc trưng hình ảnh (Re-
Identification) làm hạ tầng hỗ trợ duy trì định danh khi đối tượng di
chuyển qua nhiều camera; và học máy có giám sát kết hợp suy diễn trên đồ
thị giao lộ xây dựng từ dữ liệu bản đồ thực tế để dự đoán hướng đi dài hạn
của phương tiện. Bốn nhóm kỹ thuật này được lựa chọn vì cùng có thể vận
hành trực tiếp trên dữ liệu pixel-space của video CCTV thật, không phụ
thuộc cảm biến định vị gắn theo phương tiện mà camera giám sát giao thông
thông thường không có.

Trên cơ sở định hướng đó, đề tài xây dựng một hệ thống xử lý đồng thời
nhiều luồng camera, tự động phát hiện và gắn nhãn vi phạm giao thông cho
từng phương tiện, cho phép người vận hành đánh dấu một đối tượng nghi vấn
kèm lưu lại bằng chứng và biển số xe, đồng thời suy diễn và hiển thị lên
bản đồ thực tế hành trình nhiều khả năng nhất mà đối tượng đó sẽ tiếp tục
di chuyển qua các ngã tư phía trước. Đóng góp chính của đồ án là phần dự
đoán hành trình dài hạn: so với công trình gần nhất cùng bối cảnh giao
thông Việt Nam (Chau et al. [1], dự đoán toạ độ ngắn hạn trong một camera),
đề tài này đi xa hơn ở việc kết hợp phân loại hướng rẽ tại từng ngã tư với
suy diễn liên tiếp qua nhiều ngã tư trên một đồ thị đường thực, rồi chiếu
trực tiếp kết quả lên bản đồ. Toàn bộ hệ thống đã được vận hành thử trực
tiếp trên video CCTV thật tại một giao lộ của Hà Nội, đạt độ chính xác phân
loại hướng rẽ khoảng 60% trên một tập kiểm thử thật tách riêng khỏi dữ liệu
huấn luyện — kết quả này, cùng các số liệu định lượng chi tiết hơn cho từng
thành phần của hệ thống, được trình bày đầy đủ trong Chương 4 và Chương 5.

## 1.4 Bố cục đồ án

Phần còn lại của báo cáo đồ án tốt nghiệp này được tổ chức như sau.

Chương 2 trình bày kết quả khảo sát hiện trạng giám sát giao thông bằng
camera tại Việt Nam và phân tích các hướng giải quyết liên quan đã giới
thiệu ở trên, từ đó đặc tả chi tiết các yêu cầu chức năng dưới dạng use
case và các yêu cầu phi chức năng của hệ thống đề xuất.

Chương 3 trình bày nền tảng lý thuyết và các công nghệ được lựa chọn để
hiện thực hoá những yêu cầu đã xác định ở Chương 2, bao gồm mô hình phát
hiện đối tượng, thuật toán theo dõi đa đối tượng trong một camera, kỹ thuật
nhận diện lại hỗ trợ theo dõi xuyên camera, phương pháp suy diễn vi phạm và
sự cố, mô hình học máy phân loại hướng rẽ tại ngã tư và phương pháp xây
dựng đồ thị giao lộ từ dữ liệu bản đồ thực tế.

Chương 4 phân tích thiết kế kiến trúc của hệ thống, trình bày chi tiết quá
trình triển khai từng thành phần và đánh giá định lượng kết quả đạt được
trên dữ liệu video CCTV thật.

Chương 5 trình bày các vấn đề kỹ thuật cụ thể phát hiện được trong quá
trình xây dựng hệ thống, giải pháp đã áp dụng để khắc phục và kết quả định
lượng tương ứng, được xem là các đóng góp nổi bật của đồ án bên cạnh đóng
góp chính về dự đoán hành trình dài hạn đã nêu ở mục 1.3.

Chương 6 tổng kết lại kết quả đạt được của đồ án, đối chiếu với mục tiêu đã
đặt ra ở Chương 1, đồng thời chỉ ra các hạn chế còn tồn tại và đề xuất
hướng phát triển tiếp theo.

---

## TÀI LIỆU THAM KHẢO (Chương 1)

[1] T. Chau, D. V. Ngo, M. T. Nguyen, A. D. Nguyen-Tran, T. H. Do, "Leverage
Deep Learning Methods for Vehicle Trajectory Prediction in Chaotic Traffic,"
in *Intelligence of Things: Technologies and Applications (ICIT 2023)*,
Lecture Notes on Data Engineering and Communications Technologies, vol. 187,
Springer, Cham, 2023. DOI: 10.1007/978-3-031-46573-4_24.

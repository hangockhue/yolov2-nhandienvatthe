# Yolov2 Nhận diện vật thể

## Dataset 

- Chương trình sẽ sử dụng bộ dữ liệu COCO dataset. Download tại [đây](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5).

- Đầu vào sẽ là một vector 4 chiều (m, image_height, image_width, 3)
Với m là mini-batch size, image_height và image_width là kích thước của hình ảnh, 3 chính là số kênh màu. Ở đây là 3 kênh tương ứng với ảnh màu RGB.

- Đối với mô hình này đầu ra sẽ sử dụng 5 anchors box. Anchors box là ý tưởng của Faster RCNN. Các anchors box này được định nghĩa trước về kích thước. Thay vì dự đoán trực tiếp ra kích thước của 1 bounding box , YOLOv2 dự đoán độ sai lệch của bounting box so với các anchor box. Từ những sai lệch + anchors box => Tính ra được bounding box. Các width height nào phổ biến nhất thì chọn ra và đặt làm anchors box

- Tập dữ liệu coco chứa 80 classes đầu ra của mạng sẽ là (Px, bx, by, bh, bw, c). Với c biểu thị cho các class và coco chứa
80 lớp và nếu c là một vector 80 chiều thì mỗi boundting box sẽ là một vector 85 chiều và đầu ra của một mạng sẽ 
như sau

- (m, pro_image_height, pro_image_width, 5, 85) ở đây pro_image là 19x19 vì đang sử dụng grid 19, 5 đại diện cho số lượng anchors box, 85 sẽ đại diện cho 80 class và (Px, bx, by, bh, bw)

![alt_text](/image/Capture.PNG)

## Quá trình xử lý

### Tìm class được phát hiện bởi anchors

- Vì có 80 lớp nên phải tìm class nào được phát hiện bởi anchors box . Chúng ta cần nhân số điểm xác xuất với xác xuất lớp được xuất ra khỏi mạng. Bước lọc sẽ loại bỏ những box classes scores dưới một ngưỡng nhất định ở đây là 0.6

- Hàm xử lý ở đây là yolo_filter_boxes(). Đầu vào của hàm là:
  - box_confidence là một tensor chứa xác xuất của một đối tượng (19x19,5,1)
  - boxes là tensor chứa thông số box (bx,by,bw,bh) (19x19,5,4)
  - box_class_probs là tensor chứa xác xuất phát hiện của 80 lớp
  Ta sẽ nhân xác suất của đối tượng với xác suất của 80 lớp để tìm xác suất lớp nào là cao nhất, ngưỡng để xét ở đây là 0.6

- Sau đó qua hàm yolo_eval để sắp xếp lại để thể hiện vật thể trên hình góc, vì lúc ban đầu hình đã được resize trở thành 608x608 để xử lý.

- Trước khi đưa vào hàm yolo_head() dùng để chuyển đổi output của mô hình thành dạng có thể xử lý ở hàm tiếp theo

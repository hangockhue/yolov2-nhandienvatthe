# Yolov2 Nhận diện vật thể

## Dataset 

- Chương trình sẽ sử dụng bộ dữ liệu COCO dataset.

- Đầu vào sẽ là một vector 4 chiều (m, image_height, image_width, 3)
Với m là mini-batch size, image_height và image_width là kích thước của hình ảnh, 3 chính là số kênh màu. Ở đây là 3 kênh tương ứng với ảnh màu RGB.

- Đối với mô hình này đầu ra sẽ sử dụng 5 anchors box. Anchors box là ý tưởng của Faster RCNN. Các anchors box này được định nghĩa trước về kích thước. Thay vì dự đoán trực tiếp ra kích thước của 1 bounding box , YOLOv2 dự đoán độ sai lệch của bounting box so với các anchor box. Từ những sai lệch + anchors box => Tính ra được bounding box. Các width height nào phổ biến nhất thì chọn ra và đặt làm anchors box

- Tập dữ liệu coco chứa 80 classes đầu ra của mạng sẽ là (Px, bx, by, bh, bw, c). Với c biểu thị cho các class và coco chứa
80 lớp và nếu c là một vector 80 chiều thì mỗi boundting box sẽ là một vector 85 chiều và đầu ra của một mạng sẽ 
như sau

- (m, pro_image_height, pro_image_width, 5, 85) ở đây pro_image là 19x19 vì đang sử dụng grid 19, 5 đại diện cho số lượng anchors box, 85 sẽ đại diện cho 80 class và (Px, bx, by, bh, bw)

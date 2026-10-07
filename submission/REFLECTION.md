# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Lần chạy smoke 8 mẫu cho phán quyết PASSED với regression Δ = 0.000, còn lần chạy đầy đủ (50 mẫu target, 15 câu regression) cho FAILED với regression Δ = −0.113. Cùng một adapter, cùng một mốc, chỉ khác cỡ tập eval mà kết luận đảo chiều. Điều thứ hai: cả 6 ca sai trong 50 mẫu đều là ticket chứa cụm "Khi nào tiện" (model đoán `urgency = trung_binh` thay vì `thap`), dù cụm này xuất hiện 35 lần trong tập train, lần nào cũng nhãn `thap`.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở phần huấn luyện. Cả NB1–NB5 ở chế độ smoke chỉ mất 37 phút. Thời gian đi vào hạ tầng: kernel Colab trong VS Code không khởi động được ngay từ đầu (lỗi "Unable to get resolved server information"), việc đưa `results/` ra khỏi Colab khi `files.download` không tải được file về trong VS Code (cuối cùng phải in file thành base64 rồi giải mã ở máy local), và NB6 hỏng hai lần liên tiếp: lần một bị kill (`exit -9`) lúc lưu trọng số đã merge, lần hai báo thiếu `offload_dir` khi nạp lại base để hoán đổi adapter.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi không còn tin rằng loss thấp hơn nghĩa là cấu hình tốt hơn: `attn_only` có train loss thấp hơn `correct` (0.538 so với 0.627) nhưng điểm target hai bên bằng nhau (0.970). Tôi cũng không còn tin rằng điểm target cao là đủ để deploy: bản fine-tune thắng mốc prompt tử tế +0.205 nhưng cổng hồi quy vẫn FAILED. Và một con số trung bình đẹp (0.970) có thể che toàn bộ lỗi vào một cụm từ duy nhất.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để chẩn đoán lỗi kernel, đọc rubric và `verify.py`, giải mã `results/` từ chuỗi base64, đối chiếu từng con số trong report với file, và soạn bản nháp REPORT.md và REFLECTION.md từ số liệu thật. Nó sai ít nhất hai chỗ: lúc đầu hướng dẫn dùng `files.download` để tải kết quả, nhưng cách này không hoạt động khi chạy Colab trong VS Code; và bản sửa NB6 đầu tiên (bỏ bước lưu trọng số) chỉ giải quyết được một lỗi, phần hot-swap vẫn hỏng vì GPU chưa được giải phóng, nên phải sửa thêm lần nữa. Tôi cũng không thể chạy Colab thay tôi, nên mọi lần chạy và mọi con số vẫn phải do tôi chạy rồi dán lại.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Dựng tập eval cố định có nhãn của khách và đo mốc prompt tử tế trên base model trước, rồi mới train. Nếu mốc đó đã đủ tốt thì không fine-tune. Nếu fine-tune, tôi trộn 1–5% dữ liệu phổ thông (replay) ngay từ đầu và đo cổng hồi quy bằng tập đủ lớn, vì lab này cho thấy tập nhỏ có thể che mất đúng loại lỗi đó.

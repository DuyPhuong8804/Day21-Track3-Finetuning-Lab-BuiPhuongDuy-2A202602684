# Lab 21 — Evaluation Report

**Họ tên**: Bùi Phương Duy  **MSSV**: 2A202602684  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6 GB (sm_75, fp16)`

> Mọi con số dưới đây lấy từ file trong `results/` (`runs.csv`, `verdict.json`, `autopsy.json`, `baselines_frozen.json`, `mask_proof.json`, `token_stats.json`, `template_check.json`, `qualitative.json`).
> Lần chạy nộp bài dùng toàn bộ tập eval: 50 mẫu target, 15 câu regression, `EVAL_LIMIT` bỏ trống.

---

## 1. Setup

| | |
|---|---|
| Dataset | Mặc định của lab: 250 ticket CSKH tiếng Việt → JSON triage 4 trường (không đổi dataset) |
| Base model | Mặc định của tier T4: `unsloth/Qwen3.5-4B` (không đổi model). Lý do: giữ đúng khung đo của lab, và model 4B vừa T4 16 GB (đỉnh VRAM 8,78 GB) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (mặc định của tier). p95 đo được là **98** token *(results/token_stats.json)*, gợi ý 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch = 30 optimizer step (batch 1 × grad-accum 16) |
| Precision | fp16 + GradScaler (T4 không có bfloat16) |

**Về `max_length`:** p95 = 98 và max = 101 token, nên mọi mẫu đều ngắn hơn cả 256 lẫn 1024. Tôi giữ 1024 của tier vì không mẫu nào bị cắt, và với batch 1 không có padding thừa nên không tốn thêm VRAM đáng kể. Con số 256 chỉ là ngưỡng tối thiểu theo p95; lệch khỏi gợi ý này không ảnh hưởng kết quả.

**Template có giữ khối `<think>` không?** **Có.** `template_check.json` cho verdict *"reasoning preserved — safe to train on traces"*. Với dữ liệu triage không có trace suy luận, template render một khối `<think>` rỗng (`<think>\n\n</think>`) trước JSON. Vùng được tính loss bắt đầu từ `</think>`, rồi tới JSON và `<|im_end|>`.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token ở mẫu kiểm tra; 9014/20951 = 43,0% trên toàn tập train theo log NB3, không có trong `results/`) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Đoạn được tính loss ở chế độ `assistant-only` (giải mã ngược từ các vị trí `labels != -100`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chứng: chế độ `everything` cho `supervised 94/94 (100%)`. Khi đó loss tính cả system prompt và câu hỏi của khách, là lỗi "model viết lại câu hỏi". `assistant-only` chỉ giữ 41%, thấp xa ngưỡng mất điểm 95%.

---

## 3. Ba baseline (NB2 — mốc đóng băng)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3388.7 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1066.8 |
| (c) LoRA fine-tune | 0.970 | 0.6778 | 1.000 | 1481.4 |

n = 50 mẫu target, 15 câu regression.

**(b) có thật sự mạnh hơn (a) không?** Có: target từ 0.000 lên 0.765 và format từ 0.000 lên 1.000. Prompt naive không buộc model trả JSON nên (a) bằng 0 ở cả hai chỉ số. Tôi **không sửa** `OPTIMIZED_PROMPT` (SHA `719e74d3b6232053` khớp bản gốc, gatekeeper xác nhận), nên (b) là đối thủ nguyên bản.

**Ghi chú về thứ tự đo.** Lần đầu NB1–NB5 chạy ở chế độ smoke (`EVAL_LIMIT=8`) và NB2 đo baseline trước khi train. Với bản nộp, tôi chạy lại NB2 và NB5 trên toàn bộ tập eval sau khi adapter đã train xong. Tập eval và prompt (b) không đổi (checksum và SHA khớp), nên phép so sánh vẫn là cùng một mốc, nhưng số baseline cuối cùng được đo sau khi train chứ không phải trước. Tôi nêu điều này để trung thực.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32.464.896 | 1e-4 | 0.6265 | **0.970** | 409,1 | 8,78 |
| `attn_only` | q,v | 283 *(matched)* | 32.456.704 | 1e-4 | 0.5379 | **0.970** | 287,2 | 8,79 |
| `wrong_lr` | text-linear | 16 | 32.464.896 | 1e-5 | 1.5702 | **0.000** | 414,2 | 8,78 |
| `qlora` | text-linear | 16 | 32.464.896 | 1e-4 | 0.7058 | **0.940** | 478,8 | 3,86 |

Cả bốn run dùng cùng `max_steps = 30`. `attn_only` lệch 0,03% số tham số so với `correct` (32.456.704 so với 32.464.896), nên là phép đối chứng công bằng. Mỗi run chỉ đổi một biến: `attn_only` đổi vị trí (rank được nâng lên để giữ ngân sách), `wrong_lr` đổi LR, `qlora` đổi độ chính xác trọng số.

**4.1 — `attn_only` vs `correct`.** Hai run **hoà** ở target (0.970 cả hai, format 1.000 cả hai), dù `attn_only` chỉ gắn adapter vào 2 loại module (`q`, `v`) thay vì 12. Train loss của `attn_only` thấp hơn (0,538 so với 0,627), nên theo loss thì `attn_only` "thắng", còn theo target thì hai run bằng nhau; hai thứ tự không trùng nhau. Vì ngân sách tham số đã khớp, kết quả này cho thấy ở bài này vị trí gắn adapter không phải đòn bẩy quyết định: rank 283 ở attention bù lại được việc gắn rank 16 vào mọi lớp. Hai lưu ý để không nói quá: tập eval 50 mẫu chỉ phân biệt được chênh lệch cỡ 0,02 mỗi mẫu, và bài triage này rất dễ nên điểm chạm trần (0.970), cho nên một khác biệt nhỏ giữa hai cấu hình có thể bị che. Tôi cũng chưa đo regression của `attn_only` nên không biết nó có quên ít hơn hay không. `attn_only` huấn luyện nhanh hơn (287 s so với 409 s) và suy luận nhanh hơn (918,6 ms so với 1481,4 ms), điều này hợp lý vì chỉ có 2 loại module mang adapter, nhưng độ trễ trên T4 dùng chung dao động khá nhiều nên tôi không coi đây là kết luận chắc.

**4.2 — `wrong_lr`.** Chỉ khác LR (1e-5 so với 1e-4). Loss của `wrong_lr` giảm rất chậm: từ 2,163 xuống 1,119 sau 30 step, trong khi `correct` xuống 0,026; độ chính xác token cuối lần lượt 0,79 so với 0,996. Kết quả trên tập eval là target 0.000 và format 0.000, tức model không sinh ra JSON hợp lệ nào, và nó cũng chậm nhất (5434,3 ms). Nếu chỉ nhìn đường loss mà không biết LR, ta dễ kết luận nhầm rằng "LoRA không học được bài này" hoặc "250 mẫu quá ít", trong khi nguyên nhân là LR thang full-FT quá thấp cho LoRA và thiếu hẳn một con số. Đây là đòn bẩy lớn nhất đo được trong lab: đổi một hằng số kéo target từ 0.970 xuống 0.000.

**4.3 — `qlora`.** Tiết kiệm VRAM rõ rệt: đỉnh 3,86 GB so với 8,78 GB (giảm 56%). Cái giá là thời gian huấn luyện tăng 17% (478,8 s so với 409,1 s), độ trễ suy luận tăng 25% (1848,2 ms so với 1481,4 ms) và train loss cao hơn (0,706 so với 0,627). Target giảm từ 0.970 xuống 0.940, tức khoảng 1,5 mẫu trên 50, nằm trong vùng nhiễu của tập eval này. Vậy số đo của tôi chỉ ủng hộ một phần khuyến nghị "không dùng QLoRA cho dòng model này": trên bài này QLoRA không làm hỏng chất lượng, nhưng nó chậm hơn và vẫn nhỉnh kém. Nếu VRAM không phải nút thắt thì không có lý do để chọn nó; nếu VRAM là nút thắt (card dưới 8 GB) thì nó dùng được.

**Xếp hạng núm vặn theo ảnh hưởng đến target:** LR (0.970 → 0.000) ≫ độ chính xác trọng số (−0.030) ≈ vị trí adapter (0.000).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = −0.113` · `valid_trace_rate = 0.00`

Bản fine-tune thắng rõ ở bài toán đích: target từ 0.765 (mốc (b)) lên 0.970, và format giữ 1.000. Nhưng cổng hồi quy FAILED vì năng lực phổ thông tụt từ 0.7911 xuống 0.6778, tức −0.113, trong khi dung sai là 0.020. Nói cách khác, bản fine-tune giỏi hơn trên đúng việc được dạy nhưng đã quên một phần khả năng trả lời câu hỏi chung. Đây là hiện tượng quên thảm hoạ mà deck §6.3 đề cập; gatekeeper khuyến nghị trộn 1–5% dữ liệu phổ thông (replay) vào tập train. Hợp lý là: toàn bộ 225 mẫu train đều có cùng một định dạng ticket → JSON, 30 step ở LR 1e-4 trên mọi module tuyến tính đủ để kéo phân phối đầu ra về định dạng đó, và không có mẫu nào nhắc model giữ lại hành vi cũ. Tôi chưa kiểm chứng cơ chế này (chưa xem câu trả lời của model ở từng câu regression), nên đây là giả thuyết, không phải kết luận. Cần cân nhắc thêm hai điều: tập regression chỉ có 15 câu nên một câu sai thêm đã làm điểm nhúc nhích đáng kể (0,113 tương đương gần 1,7 câu), và `valid_trace_rate = 0.00` nghĩa là bản fine-tune không sinh khối suy luận hợp lệ, điều này nhất quán với dữ liệu train chỉ có `<think>` rỗng; tôi không đo giá trị này cho base model nên không biết có bị mất hay vốn đã bằng 0. Lần chạy smoke 8 mẫu từng cho PASSED với regression Δ = 0.000; lần đầy đủ mới lộ ra FAILED.

---

## 6. Định tính — bắt buộc có cả ca THUA

`results/qualitative.json` lưu dự đoán của bản fine-tune (c) nhưng **không lưu dự đoán của (b)**, nên cột (b) để trống và "thắng/thua" dưới đây được hiểu là so với nhãn đúng. Nhãn lấy từ `data/eval_target.jsonl`. Dự đoán trong file bị cắt cụt ở trường `sentiment`; phần nhìn thấy đủ để xác định lỗi.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 0 | …chuột không dây VN232232. Cho tôi trả lại. **Gấp.** Shop hỗ trợ tốt. | doi_tra · cao · tich_cuc | — | doi_tra · cao · tich_c… | ✅ FT đúng 4/4 (1.0) |
| 4 | …đèn bàn LED VN339109. Vỡ khi nhận. **Gấp.** | san_pham_loi · cao · trung_tinh | — | san_pham_loi · cao · trung… | ✅ FT đúng 4/4 (1.0) |
| 3 | …bình giữ nhiệt VN804124. Chưa thấy tiền. **Khi nào tiện.** | hoan_tien · **thap** · tich_cuc | — | hoan_tien · **trung_binh** · … | ❌ **FT thua** (0.75): sai urgency |
| 5 | …nồi chiên không dầu DH249548. Thiếu phụ kiện. **Khi nào tiện.** | san_pham_loi · **thap** · trung_tinh | — | san_pham_loi · **trung_binh** · … | ❌ **FT thua** (0.75): sai urgency |
| 12 | …áo khoác gió VN613097. Bị lỗi. **Khi nào tiện.** | san_pham_loi · **thap** · tich_cuc | — | san_pham_loi · **trung_binh** · … | ❌ **FT thua** (0.75): sai urgency |

Ba ca thua còn lại cùng kiểu: #39, #41, #46 (đều 0.75).

**Có mẫu chung nào ở các ca FT thua không?** Có, và rất rõ. Cả 6 ca sai (3, 5, 12, 39, 41, 46) là toàn bộ các ticket chứa cụm **"Khi nào tiện"**, và trong cả 6 ca model đoán `urgency = trung_binh` trong khi nhãn là `thap`. Đối chiếu dữ liệu: cụm này xuất hiện ở 35/250 mẫu train và cả 35 đều có nhãn `thap`; trong tập eval nó xuất hiện đúng 6 lần, và 6 lần đó chính là 6 lỗi. Nghĩa là 44 ticket còn lại đều đúng hoàn toàn, và toàn bộ phần điểm thiếu (0.970 thay vì 1.000) đến từ một cụm từ cụ thể ở một trường. Model không học trọn quy tắc "Khi nào tiện → thap" sau 30 step, dù thấy nó 35 lần.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không nên deploy** bản fine-tune này nguyên trạng. Về mục tiêu đích nó thắng rõ: target tăng từ 0.765 lên 0.970 so với base đã được prompt tử tế, format đạt 1.000 và 44/50 ticket đúng hoàn toàn. Nhưng cổng hồi quy FAILED vì năng lực phổ thông tụt từ 0.7911 xuống 0.6778, vượt xa dung sai 0.020, và việc triển khai một model đã quên cách trả lời câu hỏi chung chỉ có ý nghĩa nếu nó chỉ được dùng cho triage. Đòn bẩy thật sự trong lab này theo số đo của tôi là learning rate, không phải vị trí adapter hay rank: chỉ đổi LR từ 1e-4 xuống 1e-5 đã kéo target từ 0.970 xuống 0.000, trong khi chuyển adapter từ 12 loại module sang 2 loại (giữ nguyên ngân sách tham số) không làm điểm thay đổi, và QLoRA chỉ làm điểm giảm 0.030 trong khi giảm 56% VRAM. Phần điểm thiếu còn lại của bản fine-tune quy về một cụm từ duy nhất, "Khi nào tiện", bị gán nhầm `trung_binh`, cho thấy lỗi nằm ở dữ liệu và số step hơn là ở cấu hình LoRA. Bước tiếp theo hợp lý là trộn 1–5% dữ liệu phổ thông vào tập train để chặn việc quên, và thêm vài mẫu có nhãn `thap` đi kèm cụm đó hoặc tăng số step; sau đó đo lại bằng cùng tập eval và cùng mốc (b). Nếu cổng hồi quy vẫn FAILED sau khi thêm replay, kết luận trung thực sẽ là prompt tử tế đã đủ cho bài này.

**Ba điều tôi học được** (rút ra từ chính số đo của lab này):
1. **Lần chạy smoke 8 mẫu đã nói dối.** Ở n=8 phán quyết là PASSED với regression Δ = 0.000; ở n=50/15 nó là FAILED với regression Δ = −0.113. Một tập eval quá nhỏ không chỉ ồn, nó có thể che mất đúng loại lỗi mà cổng hồi quy sinh ra để bắt.
2. **Loss và điểm target kể hai câu chuyện khác nhau.** `attn_only` có train loss thấp hơn `correct` (0,538 so với 0,627) nhưng điểm target hoà nhau; chỉ nhìn loss thì tôi sẽ xếp hạng sai.
3. **Một lỗi lớn có thể chỉ là một cụm từ.** Toàn bộ 6 lỗi của bản fine-tune đến từ cụm "Khi nào tiện". Nhìn điểm trung bình 0.970 thì không thấy, phải mở từng ca sai và đối chiếu với nhãn và dữ liệu train mới thấy.

**Nếu có thêm 2 giờ, tôi sẽ thử:** (1) trộn 1–5% dữ liệu replay và đo lại regression, (2) lưu dự đoán của (b) theo từng mẫu để so sánh thắng/thua trực tiếp ở mục 6, (3) đo regression cho cả `attn_only` để biết vị trí adapter có ảnh hưởng đến việc quên không, và (4) chạy NB6 merge để kiểm tra độ trễ 1481 ms của bản fine-tune có giảm về gần 1067 ms của (b) không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap — **merge đã đo, hot-swap chưa có bằng chứng** (xem dưới)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

### B1 — merge và hot-swap (NB6)

`results/merge_check.json` (50 mẫu target, adapter `correct`): điểm **trước merge 0.9700**, **sau merge 0.9700**, Δ = 0.0000, ngưỡng cho phép 0.01. Assert không tụt điểm đạt.

Lần chạy NB6 đầu tiên bị kill (`exit -9`) ở bước `Writing model shards` khi lưu trọng số đã merge (khoảng 9 GB) trên Colab miễn phí; tôi nghi do hết RAM CPU nhưng chưa kiểm chứng. Vì rubric chỉ yêu cầu điểm sau merge và hot-swap, tôi bỏ dòng `merged.save_pretrained(...)` trong `notebooks/06_merge_and_serve.py` trên Colab rồi chạy lại. Do đó **trọng số đã merge không được lưu** (`adapters/merged` không tồn tại); chỉ có điểm trước/sau merge.

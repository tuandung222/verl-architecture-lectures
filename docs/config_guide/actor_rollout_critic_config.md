# Bài 3: Giải nghĩa chuyên sâu tham số Actor, Rollout và Critic

## 1. Nhóm cấu hình Actor (Actor Config)

Mô hình Actor ($\pi_\theta$) là mạng chính sách đích được tối ưu hóa. Cấu hình của nó được nạp từ tệp con [actor.yaml](file:///Users/admin/TuanDung/repos/verl/verl/trainer/config/actor/actor.yaml).

### A. Siêu tham số Batching và Quản lý Bộ nhớ
* **`ppo_mini_batch_size`** (Giá trị mặc định: `256`):
  * **Ý nghĩa**: Kích thước batch dữ liệu tối ưu hóa chính sách Actor sau khi thu thập mẫu (Rollout). Batch lớn giúp gradient ổn định nhưng đòi hỏi nhiều bộ nhớ GPU.
* **`ppo_micro_batch_size_per_gpu`** (Giá trị mặc định: `null`):
  * **Ý nghĩa**: Kích thước batch thực tế đưa vào tính toán forward/backward pass trên mỗi GPU trong một bước tính gradient. Nếu VRAM nhỏ, ta giảm thông số này và tăng số bước tích lũy gradient (Gradient Accumulation) để đạt được hiệu quả tương đương `ppo_mini_batch_size` mà không bị OOM.
* **`use_dynamic_bsz`** (Giá trị mặc định: `false`):
  * **Ý nghĩa**: Bật cơ chế thay đổi batch size động ở runtime dựa trên độ dài thực tế của chuỗi văn bản đầu vào để tối ưu hóa bộ nhớ GPU.
* **`ppo_max_token_len_per_gpu`** (Giá trị mặc định: `16384`):
  * **Ý nghĩa**: Giới hạn tổng số lượng tokens tối đa được xử lý trên mỗi GPU trong một batch huấn luyện. Giúp ngăn chặn lỗi tràn bộ nhớ khi gặp các chuỗi văn bản dài đột biến.

### B. Cấu hình Loss và tối ưu hóa toán học
* **`policy_loss.loss_mode`** (Các giá trị: `"vanilla"`, `"clip-cov"`, `"kl-cov"`, `"gpg"`):
  * **Ý nghĩa**: Chế độ tính toán hàm mất mát của PPO.
    * `"vanilla"`: Sử dụng hàm Surrogate Loss clipped tiêu chuẩn của PPO.
    * `"clip-cov"` và `"kl-cov"`: Các chế độ tối ưu hóa phân phối biến thiên giúp ổn định hóa quá trình hội tụ trên dữ liệu preference.
* **`use_kl_loss`** (Giá trị mặc định: `false`):
  * **Ý nghĩa**: Sử dụng hàm mất mát KL Divergence trực tiếp trong pha tối ưu hóa Actor thay vì trừ điểm thưởng ở môi trường. Đây là cấu hình bắt buộc khi huấn luyện bằng thuật toán **GRPO**.
* **`kl_loss_coef`** (Giá trị mặc định: `0.001`):
  * **Ý nghĩa**: Trọng số của KL Loss trong tổng hàm mất mát khi `use_kl_loss` bằng `True`.

---

## 2. Nhóm cấu hình Rollout (Rollout Engine Config)

Pha sinh mẫu (Rollout) sử dụng các engine suy luận hiệu năng cao như vLLM hoặc SGLang. Cấu hình nằm trong tệp [rollout.yaml](file:///Users/admin/TuanDung/repos/verl/verl/trainer/config/rollout/rollout.yaml).

### A. Siêu tham số Sinh mẫu (Sampling Parameters)
* **`temperature`** (Giá trị mặc định: `1.0`):
  * **Ý nghĩa**: Nhiệt độ sinh mẫu điều phối độ ngẫu nhiên của phân phối xác suất token tiếp theo. $T=1.0$ cho độ đa dạng cao, $T=0.0$ tương đương với sinh tham lam (Greedy decoding) cho kết quả cố định.
* **`n`** (Giá trị mặc định: `1`):
  * **Ý nghĩa**: Số lượng phản hồi độc lập cần sinh ra cho mỗi prompt. Đối với PPO tiêu chuẩn, `n=1`. Đối với **GRPO**, chúng ta cần sinh nhiều mẫu trên một prompt (thường `n=8` hoặc `n=16`) để thực hiện chuẩn hóa điểm thưởng trong nhóm.

### B. Quản lý tài nguyên GPU của Engine Suy luận
* **`gpu_memory_utilization`** (Giá trị mặc định: `0.5`):
  * **Ý nghĩa**: Tỷ lệ VRAM GPU tối đa được phân bổ cho vLLM hoặc SGLang làm bộ nhớ đệm KV-Cache. Trong cấu hình HybridEngine, giá trị này thường được đặt dưới `0.5` để chừa lại VRAM cho pha học của Actor/Critic.
* **`free_cache_engine`** (Giá trị mặc định: `True`):
  * **Ý nghĩa**: Giải phóng hoàn toàn bộ đệm KV-Cache của vLLM/SGLang sau khi kết thúc pha sinh mẫu, trả lại bộ nhớ GPU sạch cho pha tối ưu hóa ngược.

### C. Quản lý Tác tử Đa lượt (Multi-turn Agent)
* **`multi_turn.enable`** (Giá trị mặc định: `False`):
  * **Ý nghĩa**: Bật cơ chế tương tác đa lượt gọi công cụ (Tool calling).
* **`multi_turn.tool_config_path`**: Đường dẫn tới tệp định nghĩa Schema và cấu hình các công cụ ngoài (ví dụ Code Sandbox, Web Search).
* **`multi_turn.tokenization_sanity_check_mode`** (Các giá trị: `"strict"`, `"ignore_strippable"`, `"disable"`):
  * **Ý nghĩa**: Chế độ kiểm tra tính toàn vẹn của chuỗi token khi nối các lượt thoại đa lượt. Đảm bảo tokenization không bị sai lệch cấu trúc câu của Chat Template.

---

## 3. Nhóm cấu hình Critic (Critic Config)

Mạng Critic ($V_\phi$) chỉ được kích hoạt khi huấn luyện bằng thuật toán yêu cầu mạng đánh giá giá trị trạng thái như PPO.
* **`critic.strategy`** (Các giá trị: `"fsdp"`, `"fsdp2"`, `"megatron"`):
  * **Ý nghĩa**: Chiến lược phân mảnh mạng Critic trên GPU, thường được thiết lập đồng bộ với chiến lược của Actor để tối ưu hóa việc phân bổ tài nguyên.

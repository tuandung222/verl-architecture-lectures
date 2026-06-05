# Bài 5: Huấn luyện không cần Critic: So sánh thực nghiệm GRPO, ReMax và RLOO

## 1. Tại sao cần các giải thuật loại bỏ Critic Network?

Trong thuật toán PPO tiêu chuẩn, mạng **Critic** ($V_\phi$) chịu trách nhiệm dự đoán giá trị kỳ vọng (Value Estimation) cho mỗi trạng thái để làm điểm so sánh (Baseline) tính toán GAE. Việc duy trì và tối ưu mạng Critic này mang lại hai bất lợi rất lớn trong thực tế:
* **Chi phí VRAM khổng lồ**: Mạng Critic thường có cấu trúc và số lượng tham số tương đương với Actor (ví dụ: Actor 7B thì Critic cũng cần 7B). Điều này làm tăng gấp đôi bộ nhớ GPU cần thiết cho pha học.
* **Độ bất ổn định khi huấn luyện**: Critic cần phải hội tụ đủ nhanh để cung cấp baseline chính xác cho Actor. Nếu Critic hội tụ kém hoặc bị trôi (drift), Actor sẽ nhận được các tín hiệu Advantage sai lệch dẫn đến sụp đổ chính sách (policy collapse).

Để khắc phục, `verl` tích hợp sẵn 3 giải thuật RL không sử dụng Critic Network: **GRPO**, **ReMax**, và **RLOO**. Chúng thay thế mạng Critic bằng các phương pháp toán học ước lượng baseline trực tiếp từ nhóm các phản hồi được sinh ra.

---

## 2. So sánh bản chất toán học của các Baseline Estimators

Cả 3 giải thuật đều sử dụng thuật toán Policy Gradient nhưng khác nhau ở cách định nghĩa hàm Baseline ($b$) để tính toán Advantage ($A_i = R(y_i) - b$):

| Giải thuật | Số lượng mẫu sinh ($K$ hoặc $G$) | Công thức tính toán Advantage ($A_i$) | Bản chất của Baseline |
| :--- | :--- | :--- | :--- |
| **GRPO** | $G$ mẫu ngẫu nhiên ($G \ge 4$, thường là $8$ hoặc $16$) | $$A_i = \frac{R(y_i) - \text{mean}(R(Y))}{\text{std}(R(Y))}$$ | Trung bình cộng điểm thưởng của cả nhóm $Y$. Sử dụng độ lệch chuẩn để chuẩn hóa biên độ Advantage. |
| **ReMax** | $1$ mẫu ngẫu nhiên ($y^s$) và $1$ mẫu tham lam ($y^g$) | $$A = R(y^s) - R(y^g)$$ | Điểm số của phản hồi tham lam (Greedy decoding với $T=0$) làm baseline cố định đại diện cho khả năng hiện tại của mô hình. |
| **RLOO** | $K$ mẫu ngẫu nhiên ($K \ge 2$, thường là $4$) | $$A_i = R(y_i) - \frac{1}{K-1} \sum_{j \neq i} R(y_j)$$ | Trung bình cộng điểm thưởng của $K-1$ mẫu còn lại trong nhóm (Leave-One-Out), loại trừ chính mẫu đang xét để đảm bảo tính không chệch. |

---

## 3. So sánh cấu hình thực nghiệm và tệp kịch bản chạy trong `verl`

Chúng ta có thể đối chiếu trực tiếp cách thiết lập của 3 giải thuật này qua các file ví dụ thực tế trong repo `verl`:

### A. Kịch bản ReMax
* **Tệp chạy ví dụ**: [run_qwen2.5-7b_seq_balance.sh](file:///Users/admin/TuanDung/repos/verl/examples/remax_trainer/run_qwen2.5-7b_seq_balance.sh)
* **Cấu hình đặc trưng**:
  ```bash
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=remax \
      actor_rollout_ref.rollout.n=1 \
  ```
  * `algorithm.adv_estimator=remax`: Chỉ định ước lượng Advantage bằng thuật toán ReMax.
  * `rollout.n=1`: Chỉ cần sinh 1 mẫu ngẫu nhiên cho mỗi prompt, kết hợp với 1 lượt sinh tham lam nội bộ hệ thống.

### B. Kịch bản GRPO
* **Tệp chạy ví dụ**: [run_qwen2-7b_math.sh](file:///Users/admin/TuanDung/repos/verl/examples/grpo_trainer/run_qwen2-7b_math.sh)
* **Cấu hình đặc trưng**:
  ```bash
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      actor_rollout_ref.rollout.n=8 \
  ```
  * `algorithm.adv_estimator=grpo`: Chỉ định ước lượng Advantage bằng GRPO.
  * `rollout.n=8`: Mỗi prompt cần sinh ra 8 phản hồi độc lập để thực hiện tính toán thống kê nhóm.

### C. Kịch bản RLOO
* **Tệp chạy ví dụ**: [run_qwen2-7b.sh](file:///Users/admin/TuanDung/repos/verl/examples/rloo_trainer/run_qwen2-7b.sh)
* **Cấu hình đặc trưng**:
  ```bash
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=rloo \
      actor_rollout_ref.rollout.n=4 \
  ```
  * `algorithm.adv_estimator=rloo`: Sử dụng thuật toán RLOO.
  * `rollout.n=4`: Sinh ra 4 mẫu ngẫu nhiên để loại trừ chéo lẫn nhau khi tính toán baseline.

---

## 4. Bảng so sánh tổng hợp hiệu năng thực nghiệm

Dưới đây là bảng phân tích so sánh các đặc tính vận hành hệ thống của các giải thuật:

| Đặc tính hệ thống | PPO tiêu chuẩn | GRPO | ReMax | RLOO |
| :--- | :--- | :--- | :--- | :--- |
| **Yêu cầu Critic Network** | Có (Chiếm nhiều VRAM) | Không | Không | Không |
| **Chi phí VRAM học** | Rất cao | Thấp | Thấp | Thấp |
| **Số lượt sinh mẫu (Rollout Forward)** | 1 lượt | Nhiều lượt ($G \ge 8$) | 2 lượt (1 Random, 1 Greedy) | Vừa phải ($K \ge 4$) |
| **Tốc độ sinh mẫu (Rollout Speed)** | Nhanh nhất | Chậm (Đòi hỏi vLLM throughput lớn) | Vừa phải | Khá nhanh |
| **Độ nhạy Hyperparameter** | Rất nhạy (dễ sụp đổ nếu Critic học kém) | Ổn định tốt | Ổn định trung bình | Ổn định tốt |

---

## 5. Hướng dẫn lựa chọn giải thuật trong thực tế

* **Chọn GRPO** khi anh huấn luyện các mô hình suy luận toán học/logic (Reasoning Models) trên cụm phần cứng mạnh có hỗ trợ vLLM/SGLang song song lớn. Số lượng mẫu nhóm $G \ge 8$ giúp mô hình khám phá không gian lời giải lập luận hiệu quả.
* **Chọn ReMax** khi tài nguyên GPU cực kỳ hạn chế và anh muốn giảm thiểu tối đa số lượng mẫu sinh (chỉ cần sinh 1 mẫu ngẫu nhiên và 1 mẫu baseline tham lam).
* **Chọn RLOO** khi anh huấn luyện các tác vụ căn chỉnh hội thoại tổng quát (General Alignment - Chat) không đòi hỏi suy luận quá sâu, giúp đạt được sự cân bằng tối ưu giữa độ ổn định toán học và tốc độ huấn luyện.

# Bài 2: Giải nghĩa chuyên sâu tham số Trainer và Algorithm

## 1. Nhóm cấu hình Trainer (Trainer Config)

Nhóm cấu hình này quản lý vòng đời huấn luyện tổng thể, ghi log, checkpoint, và các thông số phân tán cơ bản. Các tham số được khai báo dưới tiền tố `trainer.*` trong [ppo_trainer.yaml](file:///Users/admin/TuanDung/repos/verl/verl/trainer/config/ppo_trainer.yaml).

### A. Quản lý vòng đời và Lưu checkpoint
* **`trainer.total_epochs`** (Giá trị mặc định: `30`):
  * **Ý nghĩa**: Tổng số chu kỳ huấn luyện đầy đủ trên toàn bộ tập dữ liệu. Trong mỗi epoch, mô hình sẽ thực hiện sinh mẫu (Rollout), tính điểm thưởng và cập nhật chính sách (Update).
* **`trainer.resume_mode`** (Các giá trị: `"auto"`, `"disable"`, `"resume_path"`):
  * **Ý nghĩa**: Phương thức khôi phục huấn luyện từ checkpoint cũ.
    * `"auto"`: Tự động khôi phục từ checkpoint mới nhất trong thư mục lưu nếu phát hiện tiến trình bị gián đoạn.
    * `"disable"`: Huấn luyện lại từ đầu, bỏ qua các checkpoint cũ.
    * `"resume_path"`: Khôi phục chính xác từ đường dẫn được định nghĩa trong `trainer.resume_from_path`.
* **`trainer.default_local_dir`** (Đường dẫn mặc định: `checkpoints/${trainer.project_name}/${trainer.experiment_name}`):
  * **Ý nghĩa**: Thư mục lưu trữ các checkpoint của mô hình Actor và Critic cục bộ trên server.

### B. Cấu hình Tài nguyên Phân tán
* **`trainer.nnodes`** (Giá trị mặc định: `1`):
  * **Ý nghĩa**: Số lượng node (máy chủ vật lý) tham gia vào cụm huấn luyện phân tán.
* **`trainer.n_gpus_per_node`** (Giá trị mặc định: `8`):
  * **Ý nghĩa**: Số lượng GPU được cấp phát trên mỗi máy chủ để Ray khởi tạo các Worker Group.
* **`trainer.critic_warmup`** (Giá trị mặc định: `0`):
  * **Ý nghĩa**: Số lượng iteration đầu tiên chỉ thực hiện cập nhật mạng Critic ($V_\phi$) mà không cập nhật mạng Actor ($\pi_\theta$). Cơ chế này giúp Critic đạt độ hội tụ tương đối trước khi Actor thay đổi chính sách, giảm phương sai gradient ban đầu.

---

## 2. Nhóm cấu hình Thuật toán (Algorithm Config)

Nhóm cấu hình này quản lý các siêu tham số toán học của giải thuật Học tăng cường, quyết định cách ước lượng Advantage và phạt KL Divergence. Được khai báo dưới tiền tố `algorithm.*` trong [ppo_trainer.yaml](file:///Users/admin/TuanDung/repos/verl/verl/trainer/config/ppo_trainer.yaml).

### A. Siêu tham số RL cơ bản
* **`algorithm.gamma`** ($\gamma$ - Giá trị mặc định: `1.0`):
  * **Ý nghĩa**: Hệ số chiết khấu cho các phần thưởng tương lai. Trong các tác vụ sinh văn bản (NLP/LLM Alignment), phần thưởng thường chỉ được chấm ở token cuối cùng hoặc toàn chuỗi (sparse reward), do đó $\gamma$ thường được cố định bằng $1.0$ để phản hồi toàn chuỗi có giá trị tương đương ở mọi bước đi.
* **`algorithm.lam`** ($\lambda$ - Giá trị mặc định: `1.0`):
  * **Ý nghĩa**: Siêu tham số điều phối sự cân bằng giữa độ chệch (bias) và phương sai (variance) trong ước lượng lợi thế GAE (Generalized Advantage Estimation). Khi $\lambda = 1.0$, GAE tương đương với phương pháp Monte Carlo thuần túy (không chệch nhưng phương sai lớn).

### B. Thuật toán ước lượng Advantage (Advantage Estimator)
* **`algorithm.adv_estimator`** (Các giá trị: `"gae"`, `"grpo"`, `"remax"`, `"rloo"`):
  * **Ý nghĩa**: Chọn bộ toán tử ước lượng lợi thế ($A_i$). Đây là tham số cốt lõi quyết định giải thuật RL nào sẽ được sử dụng:
    * `"gae"`: Sử dụng mạng Critic và ước lượng GAE (Proximal Policy Optimization - PPO tiêu chuẩn).
    * `"grpo"`: Sử dụng baseline trung bình nhóm (Group Relative Policy Optimization) - loại bỏ Critic.
    * `"remax"`: Sử dụng Greedy Baseline làm điểm so sánh - loại bỏ Critic.
    * `"rloo"`: Sử dụng trung bình cộng loại trừ chéo (Leave-One-Out) - loại bỏ Critic.

### C. Cơ chế kiểm soát độ dịch chuyển chính sách (KL Divergence Control)
* **`algorithm.use_kl_in_reward`** (Giá trị mặc định: `False`):
  * **Ý nghĩa**: Nếu bằng `True`, án phạt KL Divergence giữa chính sách hiện tại ($\pi_\theta$) và chính sách tham chiếu ($\pi_{ref}$) sẽ được cộng trực tiếp vào hàm thưởng ở cấp độ token:
    $$R_{\text{final}}(s, a) = R_{\text{env}}(s, a) - \beta D_{KL}(\pi_\theta(a|s) || \pi_{ref}(a|s))$$
* **`algorithm.kl_ctrl.type`** (Các giá trị: `"fixed"`, `"adaptive"`):
  * **Ý nghĩa**: Phương thức điều khiển hệ số KL ($\beta$):
    * `"fixed"`: Hệ số $\beta$ được cố định bằng giá trị `kl_coef`.
    * `"adaptive"`: Hệ số $\beta$ tự động tăng hoặc giảm dựa trên khoảng cách KL thực tế so với mục tiêu (`target_kl`) qua từng bước huấn luyện để duy trì mô hình trong Trust Region an toàn.

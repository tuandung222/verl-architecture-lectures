# Lộ trình học tập: Phân tích Tham số Cấu hình (Deep Dive into verl Configuration Arguments)

Chào mừng các kỹ sư hệ thống AI đến với Phần: **Phân tích Tham số Cấu hình trong thư viện verl**.

Trong quá trình vận hành và tối ưu hóa hiệu năng huấn luyện Học tăng cường (RLHF/RLAIF) cho các mô hình ngôn ngữ lớn, việc hiểu rõ ý nghĩa của các tham số cấu hình là yếu tố then chốt giúp tránh lỗi tràn bộ nhớ GPU (OOM), tăng tốc độ huấn luyện (throughput) và đảm bảo sự hội tụ thuật toán ổn định.

Thư viện `verl` sử dụng **Hydra** (một framework quản lý cấu hình mã nguồn mở của Meta) để thiết lập và tổ chức các tham số dưới dạng phân cấp. Phần này sẽ cung cấp một cẩm nang toàn diện giải nghĩa toàn bộ ý nghĩa các tham số cấu hình của verl.

---

## 🗺️ Bản đồ lộ trình học tập

Chương này gồm 3 bài học chính giải nghĩa chi tiết:

### [Bài 1: Tổng quan cấu hình Hydra của verl](roadmap_config.md)
* **Trọng tâm**: Cơ chế tổ chức tệp cấu hình phân cấp bằng Hydra.
* **Cơ chế**: Giải nghĩa cú pháp `defaults`, `_self_`, các tệp cấu hình con mặc định và cách thực hiện ghi đè tham số tại dòng lệnh (Command-line overrides).

### [Bài 2: Giải nghĩa chuyên sâu tham số Trainer và Algorithm](trainer_algo_config.md)
* **Trọng tâm**: Giải nghĩa chi tiết các tham số quản lý tiến trình huấn luyện (`trainer`) và thuật toán RL (`algorithm`).
* **Cơ chế**: Phân tích chi tiết tác động hệ thống của các tham số epoch, logging, checkpoint, hệ số suy giảm Gamma, ước lượng Advantage GAE/GRPO và các phương pháp kiểm soát độ dịch chuyển chính sách KL Divergence.

### [Bài 3: Giải nghĩa chuyên sâu tham số Actor, Rollout và Critic](actor_rollout_critic_config.md)
* **Trọng tâm**: Giải nghĩa chi tiết các tham số quản lý mạng chính sách (`actor`), pha sinh mẫu (`rollout`) và mạng đánh giá (`critic`).
* **Cơ chế**: Làm rõ các tham số tối ưu hóa bộ nhớ và hiệu suất: Batch size động, Loss Modes (vanilla/clip-cov/kl-cov), tỷ lệ KV-Cache, cập nhật trọng số và tích hợp công cụ tương tác đa lượt (Multi-turn).

---

## ⚙️ Cơ chế Quản lý Cấu hình Phân cấp của Hydra trong verl

Thư mục cấu hình cốt lõi của `verl` nằm tại [verl/trainer/config/](file:///Users/admin/TuanDung/repos/verl/verl/trainer/config/). Tệp cấu hình gốc là [ppo_trainer.yaml](file:///Users/admin/TuanDung/repos/verl/verl/trainer/config/ppo_trainer.yaml).

### 1. Phân cấp Defaults
Trong `ppo_trainer.yaml`, khối `defaults` được khai báo để Hydra tự động nạp các tệp cấu hình mặc định tương ứng với cấu trúc phần cứng/phần mềm:
```yaml
defaults:
  - model_engine: dp
  - actor@actor_rollout_ref.actor: ${model_engine}_actor
  - data@data: legacy_data
  - ref@actor_rollout_ref.ref: ${model_engine}_ref
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  - critic@critic: ${model_engine}_critic
  - reward@reward: reward
```
Cơ chế này cho phép hoán đổi nhanh động cơ phân tán (ví dụ đổi `model_engine` từ `dp` - Data Parallel sang `megatron` để Hydra tự động nạp cấu hình `megatron_actor.yaml` và `megatron_critic.yaml` thay thế cho cấu hình FSDP).

### 2. Cú pháp ghi đè dòng lệnh (Command-line Overrides)
Hydra cho phép ghi đè mọi biến số trong tệp YAML khi chạy script shell thông qua dấu chấm (`.`) mà không cần chỉnh sửa file trực tiếp. Ví dụ:
* Thay đổi thuật toán ước lượng Advantage: `algorithm.adv_estimator=grpo`
* Thay đổi số lượt sinh mẫu: `actor_rollout_ref.rollout.n=8`
* Thay đổi độ rộng của micro batch: `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8`

# Lộ trình học tập: Phân tích Thực nghiệm & Các ví dụ Huấn luyện (Deep Dive into verl Training Experiments)

Chào mừng các kỹ sư hệ thống AI và chuyên gia căn chỉnh mô hình đến với Phần: **Phân tích Thực nghiệm và Các ví dụ Huấn luyện chuyên sâu của verl**.

Trong các chương trước, chúng ta đã đi qua kiến trúc 3D-HybridEngine, mô hình Single-Controller lập trình trên Ray, và các cơ sở toán học của PPO, GRPO, ReMax. Tuy nhiên, để thực sự làm chủ một thư viện phân tán như `verl`, việc đọc hiểu cấu hình, kịch bản huấn luyện (training scripts) và cách các thành phần phần cứng/phần mềm phối hợp trong môi trường thực nghiệm là bước đi tối quan trọng.

Phần này được thiết kế để phân tích trực tiếp các kịch bản chạy trong thư mục `examples` của thư viện `verl`. Chúng ta sẽ làm rõ các use case thực tế, thiết lập hàm thưởng (Reward Model), song song hóa phần cứng, và tối ưu hóa hiệu năng tính toán.

---

## 🗺️ Bản đồ lộ trình học tập

Chương này gồm 6 bài học phân tích thực nghiệm chi tiết:

### [Bài 1: Huấn luyện PPO với mô hình phần thưởng (Reward Model) phân tán và cơ chế Colocated Critic/Reward](exp_1_ppo_colocate_rm.md)
* **Trọng tâm**: Sử dụng Neural Reward Model (Transformer Classifier) để căn chỉnh mô hình qua RLHF.
* **Cơ chế**: Kỹ thuật gom cụm Critic và Reward Model trên cùng GPU (Colocation) giúp tiết kiệm bộ nhớ và triệt tiêu băng thông truyền tải dữ liệu trên Ray.

### [Bài 2: Huấn luyện GRPO quy mô siêu lớn với Megatron-LM và Hybrid Engine](exp_2_grpo_megatron_math.md)
* **Trọng tâm**: Huấn luyện khả năng suy luận (Reasoning) quy mô lớn giống DeepSeek-R1-Zero trên dữ liệu GSM8K/MATH.
* **Cơ chế**: Sử dụng hàm thưởng dựa trên luật (Rule-based Verifier) không cần Critic. Thiết lập song song hóa ma trận TP, PP, DP phối hợp FSDP của Actor và vLLM TP của Rollout.

### [Bài 3: Huấn luyện Multi-turn Agent và Tool Calling tích hợp SGLang](exp_3_sglang_multiturn_tool.md)
* **Trọng tâm**: Huấn luyện mô hình Agent tương tác đa lượt (ReAct Agent) gọi các công cụ ngoài (Code Sandbox, Web Search API).
* **Cơ chế**: Phần thưởng định dạng (Toolcall Shaping Reward), cơ chế chèn Observation và kỹ thuật che mặt nạ (loss masking) trong pha tối ưu hóa.

### [Bài 4: Tối ưu hóa hiệu năng thu thập mẫu: Sequence Length Balancing và Fused Kernels](exp_4_performance_tuning_balance.md)
* **Trọng tâm**: Loại bỏ điểm nghẽn (bottleneck) hiệu năng lớn nhất trong RLHF ở pha Rollout/Generation.
* **Cơ chế**: Giải thuật cân bằng độ dài chuỗi (Sequence Length Balancing) trên GPU Rollout và kích hoạt các nhân tính toán tối ưu hóa (Fused Kernels).

### [Bài 5: Huấn luyện không cần Critic: So sánh thực nghiệm GRPO, ReMax và RLOO](exp_5_critic_free_comparison.md)
* **Trọng tâm**: Đối chiếu 3 giải thuật RL không sử dụng Critic Network của `verl`.
* **Cơ chế**: Phân tích toán học và cấu hình thực nghiệm của GRPO (Group Baseline), ReMax (Greedy Baseline) và RLOO (Leave-One-Out Baseline) để lựa chọn giải thuật phù hợp cho dự án hạn chế tài nguyên.

### [Bài 6: Hướng dẫn chuẩn bị Code & Setup môi trường thực thi trên GPU Server](exp_6_gpu_server_setup.md)
* **Trọng tâm**: Cài đặt môi trường và kiểm thử khói (Smoke Test) trên GPU Server vật lý.
* **Cơ chế**: Quá trình cài đặt gói thư viện tương thích (PyTorch, FA-2, vLLM/SGLang), khởi chạy Ray cluster và chuẩn bị tập dữ liệu huấn luyện dạng Parquet.

---

## 🎯 Mục tiêu đầu ra
Sau khi hoàn thành chương học này, học viên có khả năng:
1. Đọc hiểu và tùy chỉnh mọi tệp kịch bản huấn luyện (`.sh`) trong thư mục `examples` của `verl`.
2. Tự thiết lập cấu hình phân chia GPU song song hóa (TP/PP/DP/FSDP) tối ưu cho các mô hình có kích thước khác nhau.
3. Thiết kế các hàm thưởng lai (Hybrid Reward Models) kết hợp cả Neural Network và Rule-based Verifier.
4. Debug các lỗi thường gặp về tràn bộ nhớ GPU (OOM) hoặc nghẽn cổ chai truyền thông (communication bottleneck) trong quá trình huấn luyện RLHF thực tế.

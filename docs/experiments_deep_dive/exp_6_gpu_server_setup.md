# Bài 6: Hướng dẫn chuẩn bị Code & Setup môi trường thực thi trên GPU Server

## 1. Yêu cầu hệ thống và Chuẩn bị Phần cứng (Hardware Prerequisites)

Để huấn luyện các mô hình ngôn ngữ lớn (LLM) bằng phương pháp Học tăng cường (RLHF/RLAIF) thông qua thư viện `verl`, hệ thống phần cứng của anh cần đáp ứng các điều kiện tối thiểu sau:
* **Hệ điều hành**: Linux (khuyến nghị Ubuntu 20.04 hoặc 22.04 LTS).
* **Kiến trúc GPU**: NVIDIA Ampere (A100, A800), Hopper (H100, H800) hoặc các dòng GPU hỗ trợ đầy đủ tập lệnh Tensor Core thế hệ mới.
* **Driver NVIDIA**: Phiên bản $\ge 525.60.13$ tương thích với CUDA Toolkit $\ge 12.1$.

---

## 2. Cài đặt Môi trường và Dependencies (Environment Installation)

Chúng ta sẽ cài đặt môi trường sạch sử dụng Conda để tránh xung đột gói thư viện giữa các phiên bản PyTorch, vLLM và SGLang.

### Bước 2.1: Khởi tạo Conda Environment
```bash
conda create -n verl_env python=3.10 -y
conda activate verl_env
```

### Bước 2.2: Cài đặt PyTorch và CUDA Runtime
`verl` yêu cầu PyTorch có hỗ trợ CUDA phân tán. Thực hiện cài đặt thông qua Pip:
```bash
pip3 install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 2.3: Cài đặt FlashAttention-2
FlashAttention-2 là nhân tối ưu cực kỳ quan trọng giúp giảm độ phức tạp tính toán Attention và tiết kiệm VRAM GPU:
```bash
pip install flash-attn --no-build-isolation
```

### Bước 2.4: Cài đặt các Inference Engines tương thích
Để chạy pha Rollout (sinh mẫu), ta cài đặt vLLM hoặc SGLang. 

> [!WARNING]
> **Tránh phiên bản lỗi:**
> Hãy tránh sử dụng vLLM phiên bản `0.7.x` do chứa các lỗi rò rỉ bộ nhớ (memory leaks) dẫn đến lỗi OOM đột ngột. Khuyến nghị cài đặt vLLM phiên bản $\ge 0.8.2$.

```bash
# Cài đặt vLLM
pip install vllm==0.8.2

# Cài đặt SGLang (yêu cầu cho Multi-turn Tool Calling)
pip install "sglang[all]>=0.4.3" --find-links https://flashinfer.ai/whl/cu121/torch2.4/simple
```

### Bước 2.5: Cài đặt thư viện verl ở chế độ Editable
Editable mode giúp các thay đổi mã nguồn cục bộ được cập nhật trực tiếp mà không cần cài đặt lại thư viện:
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

---

## 3. Khởi chạy Cụm phân tán Ray (Ray Cluster Initialization)

`verl` sử dụng Ray để quản lý vòng đời của các Actor/Critic/Rollout Worker Groups. Tiến trình Ray phải được khởi chạy thủ công trên server trước khi nộp tác vụ huấn luyện.

### A. Triển khai trên máy chủ đơn (Single-Node Setup)
Nếu server của anh là một node duy nhất chứa nhiều GPU (ví dụ máy chủ 8x A100):
```bash
# Đảm bảo tắt hoàn toàn các tiến trình Ray cũ nếu có
ray stop

# Khởi chạy node đầu não (Head Node)
ray start --head --port=6379
```

### B. Triển khai trên nhiều máy chủ mạng (Multi-Node Setup)
Nếu cụm máy chủ gồm máy chủ Master (Head Node) và các máy chủ phụ (Worker Nodes):
* **Trên Head Node (Master)**:
  ```bash
  ray start --head --port=6379 --node-ip-address=<MASTER_IP>
  ```
* **Trên mỗi Worker Node**:
  ```bash
  ray start --address='<MASTER_IP>:6379'
  ```

---

## 4. Chuẩn bị và Tiền xử lý dữ liệu (Data Preprocessing)

`verl` nạp dữ liệu huấn luyện thông qua định dạng bảng **Apache Parquet** để tăng tốc độ I/O. Mỗi hàng dữ liệu Parquet cần chứa các cột bắt buộc: `prompt` (câu hỏi) và `answer` (đáp án đúng làm Ground Truth).

Thư viện cung cấp sẵn các script tải và chuẩn bị dữ liệu trong thư mục [examples/data_preprocess/](file:///Users/admin/TuanDung/repos/verl/examples/data_preprocess):

```bash
# Di chuyển tới thư mục tiền xử lý dữ liệu
cd examples/data_preprocess

# Tải và định dạng lại tập dữ liệu GSM8K
python3 gsm8k.py --local_dir ~/data/gsm8k
```
Sau khi chạy xong, trong thư mục `~/data/gsm8k` sẽ xuất hiện hai tệp:
* `train.parquet`: Chứa dữ liệu huấn luyện.
* `test.parquet`: Chứa dữ liệu kiểm thử validation.

---

## 5. Chạy thử nghiệm khói (Smoke Test)

Để xác nhận hệ thống phần cứng, CUDA, Ray và thư viện `verl` liên thông hoàn chỉnh, chúng ta tiến hành chạy thử nghiệm khói (Smoke Test) bằng một mô hình nhỏ (Qwen2.5-3B) sử dụng thuật toán GRPO LoRA trên tập dữ liệu GSM8K vừa tải về:

```bash
# Quay lại thư mục gốc của verl
cd ../..

# Thực thi kịch bản huấn luyện GRPO LoRA
bash examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
```

### Các chỉ số cần theo dõi để xác nhận thành công:
1. **Ray Dashboard** (mặc định tại `http://localhost:8265`) hiển thị các Ray Actors đang hoạt động.
2. Lệnh `nvidia-smi` hiển thị tiến trình Python đang phân bổ VRAM và hoạt động tính toán trên toàn bộ GPU được cấp phát.
3. Nhật ký Terminal bắt đầu in ra các thông số huấn luyện (Step, Loss, Mean Reward, KL Divergence).

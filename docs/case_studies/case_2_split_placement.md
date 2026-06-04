---
sidebar_position: 3
sidebar_label: "Case 2: Tối ưu phần cứng với Split Placement"
---

# Case Study 2: Lập lịch tối ưu phần cứng với Split Placement

Khi kích thước mô hình ngôn ngữ tăng lên (ví dụ từ 7B sang 70B hoặc mô hình MoE lớn hơn), việc colocated (đặt chung) cả 4 mô hình lên cùng một tập GPU sẽ lập tức làm sập hệ thống do lỗi tràn bộ nhớ VRAM (OOM). 

Để vượt qua giới hạn vật lý này, `verl` hỗ trợ cơ chế **Split Placement** – tách biệt tài nguyên GPU cho các vai trò khác nhau.

---

## 1. Nguyên lý phân tách tài nguyên GPU

Thay vì ép tất cả GPU phải tải và chạy luân phiên 4 mô hình (Actor, Reference, Critic, Reward), ta chia cụm GPU thành các nhóm chuyên biệt:

* **Nhóm 1 (GPU 0..3)**: Dành riêng cho **Actor và Rollout (vLLM)**. Nhóm này cần VRAM trống tối đa để chứa bộ đệm KV Cache phục vụ sinh mẫu câu trả lời dài.
* **Nhóm 2 (GPU 4..7)**: Dành riêng cho **Critic và Reward Model**. Nhóm này tập trung vào tính toán lan truyền ngược để cập nhật hàm giá trị (Value update).

```
   Cụm 8 GPU Vật lý:
   ┌──────────┬──────────┬──────────┬──────────┐
   │  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │  ◄── Nhóm Actor & Rollout (vLLM TP4)
   └──────────┴──────────┴──────────┴──────────┘
   ┌──────────┬──────────┬──────────┬──────────┐
   │  GPU 4   │  GPU 5   │  GPU 6   │  GPU 7   │  ◄── Nhóm Critic & Reward (FSDP4)
   └──────────┴──────────┴──────────┴──────────┘
```

---

## 2. Thiết lập trong mã nguồn `main_ppo_split.py`

Trong mã nguồn thực tế của `verl` tại `examples/split_placement/main_ppo_split.py`, việc phân nhóm tài nguyên được quản lý thông qua hai đối tượng `RayResourcePool`:

```python
# Trích đoạn logic trong main_ppo_split.py
from verl.single_controller.ray import RayResourcePool, ResourcePoolManager

# 1. Khởi tạo 2 nhóm tài nguyên độc lập
actor_rollout_pool = RayResourcePool(
    process_on_nodes=[4], # Chạy 4 tiến trình trên 1 node (sử dụng GPU 0..3)
    n_gpus_per_node=4
)

critic_pool = RayResourcePool(
    process_on_nodes=[4], # Chạy 4 tiến trình khác trên 1 node (sử dụng GPU 4..7)
    n_gpus_per_node=4
)

# 2. Quản lý chung bằng ResourcePoolManager
resource_pool_manager = ResourcePoolManager(
    resource_pool_dict={
        'actor_rollout': actor_rollout_pool,
        'critic': critic_pool
    }
)
```

Khi khởi tạo `RayPPOTrainer`, ta truyền đối tượng `resource_pool_manager` này vào. Hệ thống Single-Controller của `verl` sẽ tự động phân phối các Actor Worker lên nhóm tài nguyên `actor_rollout` và Critic Worker lên nhóm `critic`.

---

## 3. Giao tiếp qua bộ nhớ chia sẻ Ray Plasma Store

Vì các mô hình Actor và Critic giờ đây nằm trên các GPU hoàn toàn khác nhau (thậm chí có thể ở các máy chủ vật lý khác nhau trong cụm), chúng không thể truyền dữ liệu bằng con trỏ bộ nhớ hay NCCL All-Gather trực tiếp trong cùng một GPU Group.

**Giải pháp giao tiếp của verl:**

1. **Serialization (Tuần tự hóa)**: Sau khi Rollout hoàn thành, Actor Worker đóng gói dữ liệu đầu ra thành `DataProto`, tự động chuyển đổi các Tensor thành dạng mảng Byte liên tục.
2. **Ray Plasma Store**: Dữ liệu byte được đẩy vào **Ray Object Store** (bộ đệm bộ nhớ chia sẻ Plasma chạy ngầm trong cụm). Ray sẽ tự động định tuyến và truyền tải gói dữ liệu này qua mạng LAN (sử dụng gRPC) tới node chứa Critic.
3. **Deserialization (Giải tuần tự hóa)**: Critic Worker nhận mảng byte từ Ray Object Store, tái dựng lại thành đối tượng `DataProto` và nạp lên bộ nhớ GPU của mình để chạy pha huấn luyện.

---

## 4. So sánh hiệu năng thực tế

| Tiêu chí | Colocated Placement | Split Placement |
| :--- | :--- | :--- |
| **Kích thước mô hình tối đa** | Bị giới hạn (Ví dụ: tối đa 7B trên 8x A100 80G) | Lớn hơn nhiều (Ví dụ: 70B trên cùng cụm nhờ phân chia VRAM) |
| **Tối ưu hóa KV Cache** | Bị bóp nghẹt do VRAM phải chia sẻ cho Critic | Đạt mức tối đa vì vLLM được sở hữu riêng cụm GPU |
| **Độ trễ truyền dữ liệu** | Cực thấp (NCCL sao chép trực tiếp GPU-to-GPU) | Trung bình (Tốn chi phí truyền dữ liệu qua Ray Object Store) |
| **Độ phức tạp cấu hình** | Rất đơn giản | Phức tạp hơn (Cần cấu hình Placement Groups trên Ray cluster) |

## 💡 Kết luận

Split Placement là vũ khí hạng nặng giúp `verl` mở rộng khả năng huấn luyện lên quy mô công nghiệp:
* Cho phép các nhóm phát triển chạy mô hình lớn vượt quá dung lượng một node GPU.
* Tạo điều kiện tích hợp hoàn hảo với các mô hình Reward siêu lớn (như các mô hình GPT-4 đánh giá tự động) mà không sợ làm sập hệ thống huấn luyện Actor.

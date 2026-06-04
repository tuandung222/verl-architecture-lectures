---
sidebar_position: 6
sidebar_label: "Bài 5: Đại số Resharding & Độ phức tạp Giao tiếp"
---

# Bài 5: Đại số phân mảnh (Resharding Algebra) & Độ phức tạp Giao tiếp

Kiến trúc **3D-HybridEngine** của `verl` giải quyết sự bất tương xứng song song bằng cách biến đổi (resharding) trọng số mô hình chéo giữa cấu hình huấn luyện (FSDP) và suy luận (Tensor Parallelism). Bài viết này phân tích sâu về mặt đại số tuyến tính mô hình phân mảnh trọng số, cách lập bản đồ ánh xạ và độ phức tạp truyền thông NCCL đi kèm.

---

## 1. Biểu diễn đại số của hai trạng thái phân mảnh: FSDP vs Tensor Parallelism

Xét một ma trận trọng số (weight tensor) $W$ của một lớp tuyến tính trong mạng Transformer có kích thước $[M, N]$ (ví dụ: lớp chiếu của Attention).

### 1.1. Phân mảnh tham số trong PyTorch FSDP (ZeRO-3)
Trong FSDP với cụm gồm $P$ GPU, tham số $W$ được làm phẳng thành mảng 1D có kích thước $MN$ và băm nhỏ thành $P$ phần bằng nhau:

$$W_{\text{FSDP}}^i = W_{\text{flat}}\left[ i \cdot \frac{MN}{P} : (i+1) \cdot \frac{MN}{P} \right]$$

Mỗi GPU Rank $i$ chỉ lưu giữ đúng $1/P$ phần của ma trận trọng số đã làm phẳng. Cấu trúc này không quan tâm đến ngữ nghĩa hình học (hàng hay cột) của ma trận gốc.

### 1.2. Phân mảnh ma trận trong Tensor Parallelism (TP)
Ngược lại, Tensor Parallelism (ví dụ: Megatron-LM hay vLLM) chia nhỏ ma trận $W$ theo ngữ nghĩa hình học để tính toán song song:

* **Column Parallel (Song song theo Cột)**: Chia ma trận $W$ dọc theo chiều cột thành $TP\_size$ phần:
  $$W = \left[ W_{\text{TP}}^0, W_{\text{TP}}^1, \dots, W_{\text{TP}}^{TP\_size-1} \right]$$
  Mỗi GPU Rank $i$ lưu trữ $W_{\text{TP}}^i$ kích thước $\left[ M, \frac{N}{TP\_size} \right]$.
* **Row Parallel (Song song theo Hàng)**: Chia ma trận $W$ ngang theo chiều hàng:
  $$W = \left[ W_{\text{TP}}^0 ; W_{\text{TP}}^1 ; \dots ; W_{\text{TP}}^{TP\_size-1} \right]$$
  Mỗi GPU Rank $i$ lưu trữ $W_{\text{TP}}^i$ kích thước $\left[ \frac{M}{TP\_size}, N \right]$.

---

## 2. Đại số ánh xạ trọng số (Resharding Algebra)

Để chuyển đổi từ định dạng FSDP sang TP (trước pha Rollout) và ngược lại (sau pha Rollout), `verl` phải thực hiện phép biến đổi ma trận:

$$f: \{W_{\text{FSDP}}^0, \dots, W_{\text{FSDP}}^{P-1}\} \longrightarrow \{W_{\text{TP}}^0, \dots, W_{\text{TP}}^{TP-1}\}$$

Phép biến đổi này diễn ra qua 4 bước đại số:

1. **All-Gather (Gom nhóm)**: Chạy truyền thông NCCL All-Gather trên tập hợp GPU để tái dựng lại mảng 1D đầy đủ từ các mảnh FSDP:
   $$W_{\text{flat}} = \text{All-Gather}\left(W_{\text{FSDP}}^i\right)$$
2. **Reshape (Định hình lại)**: Khôi phục mảng 1D về cấu trúc ma trận 2D gốc:
   $$W = \text{Reshape}\left(W_{\text{flat}}, [M, N]\right)$$
3. **Slice (Cắt lát)**: Cắt ma trận $W$ theo chiều cột hoặc hàng tùy thuộc vào loại lớp tuyến tính để tạo ra tập hợp các mảnh TP:
   $$W_{\text{TP}}^j = \text{Slice}\left(W, \text{dim}, j\right)$$
4. **Scatter/Placement (Phân phát)**: Nạp các mảnh $W_{\text{TP}}^j$ lên các GPU Rank tương ứng của vLLM Engine.

---

## 3. Phân tích độ phức tạp truyền thông NCCL (Communication Complexity)

Các phép toán giao tiếp chéo GPU tiêu tốn rất nhiều thời gian. Chúng ta tính toán dung lượng dữ liệu cần truyền tải (Communication Volume) như sau:

### 3.1. Độ phức tạp All-Gather (Cho Resharding Trọng số)
Sử dụng thuật toán Ring All-Gather với kích thước thông điệp trọng số là $S$ bytes và cụm gồm $P$ GPU, dung lượng dữ liệu truyền qua card mạng của mỗi GPU là:

$$\text{Comm}_{\text{All-Gather}} = \frac{P-1}{P} \cdot S$$

Vì bước Resharding trọng số này chỉ chạy một lần duy nhất trước mỗi pha Rollout, chi phí này là cực nhỏ so với thời gian tính toán của toàn bộ epoch.

### 3.2. Độ phức tạp All-to-All (Cho định tuyến MoE Expert Routing)
Trong huấn luyện MoE lớn (như DeepSeek 671B), khi định tuyến $N_{\text{tokens}}$ token (mỗi token có chiều ẩn là $D$) qua $P$ GPU để đến các Experts tương ứng, ta sử dụng phép toán All-to-All. Dung lượng dữ liệu truyền qua card mạng mỗi GPU là:

$$\text{Comm}_{\text{All-to-All}} = \frac{P-1}{P} \cdot N_{\text{tokens}} \cdot D$$

Vì All-to-All phải chạy ở **mọi lớp MoE** và **mọi bước forward/backward**, nó trở thành cổ chai truyền thông lớn. `verl` tối ưu hóa bằng cách kết hợp Sequence Parallelism (DeepSpeed Ulysses) để phân mảnh chiều dọc token trước khi chạy All-to-All, giúp giảm thiểu nghẽn I/O.

---

## 4. Mã giả Sharding Manager (FSDP sang vLLM TP)

Mã giả dưới đây mô phỏng lớp `FSDPVLLMShardingManager` quản lý NCCL Group và ánh xạ trọng số FSDP sang vLLM TP:

```python
import torch
import torch.distributed as dist

class FSDPVLLMShardingManager:
    def __init__(self, fsdp_group, vllm_tp_group):
        self.fsdp_group = fsdp_group
        self.vllm_tp_group = vllm_tp_group

    def reshard_fsdp_to_vllm(self, fsdp_flat_param: torch.Tensor, original_shape: tuple, is_col_parallel: bool = True) -> torch.Tensor:
        """Chuyển đổi Flat Parameter của FSDP sang Tensor sliced của vLLM TP"""
        # 1. Khởi tạo tensor đệm chứa toàn bộ trọng số flat
        world_size = dist.get_world_size(self.fsdp_group)
        gathered_flat = torch.zeros(fsdp_flat_param.numel() * world_size, 
                                    dtype=fsdp_flat_param.dtype, 
                                    device=fsdp_flat_param.device)
        
        # 2. Gom toàn bộ trọng số FSDP chéo GPU
        dist.all_gather_into_tensor(gathered_flat, fsdp_flat_param, group=self.fsdp_group)
        
        # 3. Định hình lại về ma trận 2D
        full_weight = gathered_flat[:original_shape[0] * original_shape[1]].view(original_shape)
        
        # 4. Cắt lát ma trận theo cơ chế TP
        tp_size = dist.get_world_size(self.vllm_tp_group)
        tp_rank = dist.get_rank(self.vllm_tp_group)
        
        if is_col_parallel:
            # Song song theo cột: cắt chiều thứ 1 (N)
            col_size = original_shape[1] // tp_size
            tp_weight = full_weight[:, tp_rank * col_size : (tp_rank + 1) * col_size]
        else:
            # Song song theo hàng: cắt chiều thứ 0 (M)
            row_size = original_shape[0] // tp_size
            tp_weight = full_weight[tp_rank * row_size : (tp_rank + 1) * row_size, :]
            
        return tp_weight.contiguous()
```

## 💡 Liên hệ thực tế mã nguồn verl
Hiện thực tối ưu CUDA-level của cấu trúc này nằm tại thư mục [verl/workers/sharding_manager/](file:///Users/admin/TuanDung/repos/verl/verl/workers/sharding_manager) (đặc biệt là tệp `fsdp_vllm.py` và `megatron_vllm.py`). `verl` viết trực tiếp các hàm trỏ chéo bộ nhớ để giảm tối đa chi phí cấp phát bộ nhớ động của PyTorch, giúp quá trình Resharding diễn ra cực kỳ nhanh chóng.

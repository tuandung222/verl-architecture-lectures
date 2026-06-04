---
sidebar_position: 10
sidebar_label: "Bài 8: Thực hành: Tự viết Toy RLHF Pipeline với Single-Controller"
---

# Bài 8: Thực hành: Tự viết Toy RLHF Pipeline với Single-Controller

Không có cách nào hiểu sâu sắc kiến trúc hệ thống của `verl` tốt hơn việc **tự tay xây dựng một phiên bản thu nhỏ (Toy Implementation)** của nó. 

Trong bài học thực hành này, chúng ta sẽ viết một chương trình Python hoàn chỉnh, mô phỏng lại toàn bộ cơ chế cốt lõi của `verl`: Mô hình lập trình Single-Controller (Bộ điều phối tuần tự), kiến trúc WorkerGroup (quản lý song song qua Threading), cấu trúc đóng gói tin nhắn `DataProto`, và một chu kỳ huấn luyện toán học đơn giản.

---

## 1. Thiết kế của Toy RLHF Engine

Hệ thống tối giản của chúng ta gồm các thành phần sau:

1. **`ToyDataProto`**: Đóng gói các tensor dữ liệu và metadata. Hỗ trợ cắt nhỏ (split/chunk) dữ liệu và gộp (concat) dữ liệu.
2. **`ToyWorker`**: Đại diện cho một tiến trình Worker từ xa (chạy song song trên các thread khác nhau), chịu trách nhiệm tính toán lan truyền xuôi/ngược và cập nhật trọng số.
3. **`ToyWorkerGroup`**: Đóng vai trò là proxy (đại diện) của Driver, tự động chia nhỏ dữ liệu (`dispatch`) và gom kết quả (`collect`) khi Driver gọi hàm.
4. **Driver Loop (Single-Controller)**: Chạy một luồng tuần tự duy nhất để huấn luyện mạng.

---

## 2. Hiện thực hóa mã nguồn Python (Chạy trực tiếp được)

Dưới đây là file mã nguồn hoàn chỉnh. Bạn có thể lưu lại thành một file Python (ví dụ: `toy_verl_pipeline.py`) và chạy trực tiếp để kiểm chứng hoạt động.

```python
import time
import threading
from typing import List, Dict, Callable
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. Định nghĩa ToyDataProto (Giao thức trao đổi dữ liệu)
# =====================================================================
class ToyDataProto:
    def __init__(self, batch: Dict[str, torch.Tensor], non_tensor: Dict[str, list] = None):
        self.batch = batch  # Chứa các tensor có cùng chiều batch size (dim 0)
        self.non_tensor = non_tensor if non_tensor else {}

    def __len__(self):
        random_key = list(self.batch.keys())[0]
        return self.batch[random_key].shape[0]

    def chunk(self, n: int) -> List['ToyDataProto']:
        """Chia nhỏ DataProto thành n phần bằng nhau theo chiều Batch Size"""
        chunked_batches = [{} for _ in range(n)]
        for key, tensor in self.batch.items():
            splits = torch.chunk(tensor, chunks=n, dim=0)
            for i, split in enumerate(splits):
                chunked_batches[i][key] = split

        chunked_non_tensors = [{} for _ in range(n)]
        for key, val_list in self.non_tensor.items():
            # Chia nhỏ danh sách phi tensor
            size = len(val_list)
            chunk_size = size // n
            for i in range(n):
                chunked_non_tensors[i][key] = val_list[i * chunk_size : (i + 1) * chunk_size]

        return [ToyDataProto(chunked_batches[i], chunked_non_tensors[i]) for i in range(n)]

    @staticmethod
    def concat(protos: List['ToyDataProto']) -> 'ToyDataProto':
        """Gom nhiều DataProto nhỏ thành một DataProto lớn duy nhất"""
        concat_batch = {}
        keys = protos[0].batch.keys()
        for key in keys:
            concat_batch[key] = torch.cat([p.batch[key] for p in protos], dim=0)

        concat_non_tensor = {}
        non_tensor_keys = protos[0].non_tensor.keys()
        for key in non_tensor_keys:
            concat_non_tensor[key] = []
            for p in protos:
                concat_non_tensor[key].extend(p.non_tensor[key])

        return ToyDataProto(concat_batch, concat_non_tensor)


# =====================================================================
# 2. Định nghĩa Decorator @toy_register (Dispatch & Collect tự động)
# =====================================================================
def toy_register():
    """Decorator tự động hóa quá trình cắt nhỏ và gom dữ liệu giữa Driver và Workers"""
    def decorator(func: Callable):
        def wrapper(worker_group: 'ToyWorkerGroup', data_proto: ToyDataProto, *args, **kwargs):
            # 1. Cắt dữ liệu thành các phần tương đương với số lượng workers
            n_workers = len(worker_group.workers)
            chunks = data_proto.chunk(n_workers)

            # 2. Gửi bất đồng bộ tới các workers tính toán song song
            threads = []
            results = [None] * n_workers

            def run_worker(idx, worker, chunk_data):
                # Gọi hàm thực tế trên worker
                bound_method = getattr(worker, func.__name__)
                results[idx] = bound_method(chunk_data, *args, **kwargs)

            for i, worker in enumerate(worker_group.workers):
                t = threading.Thread(target=run_worker, args=(i, worker, chunks[i]))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # 3. Thu thập và ghép các kết quả trả về thành một DataProto duy nhất
            return ToyDataProto.concat(results)
        return wrapper
    return decorator


# =====================================================================
# 3. Định nghĩa ToyWorker (Mô phỏng GPU Worker tính toán cục bộ)
# =====================================================================
class ToyWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        # Mạng neural đồ chơi đại diện cho Actor Model
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def generate(self, data: ToyDataProto) -> ToyDataProto:
        """Thực thi suy luận (Generation)"""
        inputs = data.batch['inputs']
        with torch.no_grad():
            outputs = self.model(inputs)
        print(f"    [Worker {self.worker_id}] Generate xong batch size: {inputs.shape[0]}")
        return ToyDataProto(batch={'outputs': outputs})

    def update_model(self, data: ToyDataProto) -> ToyDataProto:
        """Thực thi huấn luyện lan truyền ngược (Training/Update)"""
        inputs = data.batch['inputs']
        targets = data.batch['targets']
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        print(f"    [Worker {self.worker_id}] Train Step: Loss = {loss.item():.4f}")
        return ToyDataProto(batch={'loss': torch.tensor([loss.item()])})


# =====================================================================
# 4. Định nghĩa ToyWorkerGroup (Proxy đại diện cho cụm Workers)
# =====================================================================
class ToyWorkerGroup:
    def __init__(self, n_workers: int):
        self.workers = [ToyWorker(worker_id=i) for i in range(n_workers)]

    @toy_register()
    def generate(self, data: ToyDataProto) -> ToyDataProto:
        pass  # Phần xử lý thực tế được decorator @toy_register đảm nhận

    @toy_register()
    def update_model(self, data: ToyDataProto) -> ToyDataProto:
        pass  # Phần xử lý thực tế được decorator @toy_register đảm nhận


# =====================================================================
# 5. Luồng Driver tuần tự (Single-Controller Driver)
# =====================================================================
def main_driver():
    print("=== Khởi tạo Cụm WorkerGroup với 2 GPU Workers ===")
    worker_group = ToyWorkerGroup(n_workers=2)

    # Dữ liệu huấn luyện đồ chơi (Batch size = 8)
    toy_inputs = torch.randn(8, 10)
    toy_targets = torch.randn(8, 2)
    prompt_list = [f"prompt_id_{i}" for i in range(8)]

    # Đóng gói dữ liệu đầu vào thành DataProto
    input_proto = ToyDataProto(
        batch={'inputs': toy_inputs, 'targets': toy_targets},
        non_tensor={'prompts': prompt_list}
    )

    print("\n--- PHA 1: PHÁT SINH MẪU (ROLLOUT PHASE) ---")
    # Driver gọi hàm tuần tự như chạy single-process, dữ liệu tự động băm nhỏ và gom lại
    output_proto = worker_group.generate(input_proto)
    print(f"-> Driver nhận về kết quả sinh mẫu, tổng Batch Size: {len(output_proto)}")
    print(f"   Outputs Tensor Shape: {output_proto.batch['outputs'].shape}")

    print("\n--- PHA 2: HUẤN LUYỆN (UPDATE/TRAINING PHASE) ---")
    # Tích hợp thêm outputs vào dữ liệu đầu vào để chuẩn bị cập nhật
    input_proto.batch['outputs'] = output_proto.batch['outputs']
    
    # Driver phát lệnh huấn luyện song song
    loss_proto = worker_group.update_model(input_proto)
    mean_loss = loss_proto.batch['loss'].mean().item()
    print(f"-> Driver hoàn thành huấn luyện step 1. Loss trung bình cụm: {mean_loss:.4f}")

if __name__ == '__main__':
    main_driver()
```

---

## 💡 Đúc kết từ Bài học Thực hành

Chương trình đồ chơi trên đã mô phỏng hoàn hảo cách hoạt động bên trong của `verl`:
1. **Trong mắt Driver**: Lệnh gọi `worker_group.generate(input_proto)` diễn ra rất tự nhiên và liền mạch trên một luồng chính duy nhất. Không hề có mã quản lý song song phức tạp nào xuất hiện ở Driver.
2. **Trong mắt Worker**: Lớp `ToyWorker` chỉ cần tập trung làm nhiệm vụ tính toán cục bộ trên phần dữ liệu nhỏ được phân phối (giống như FSDP Worker chạy trên từng GPU riêng).
3. **Cầu kết nối `@toy_register`**: Đảm nhiệm vai trò của sharding manager và decorator của verl, tự động cắt lát dữ liệu `DataProto` theo batch dim và gom lại sau khi tính toán xong.

Chúc mừng bạn đã hoàn thành trọn vẹn chuỗi bài học phân tích kiến trúc của thư viện **verl**! Hy vọng kiến thức này sẽ giúp bạn thiết kế và vận hành thành công các hệ thống huấn luyện Alignment LLM hiệu năng cao.

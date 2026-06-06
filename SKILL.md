---
name: docusaurus-curriculum-deployment
description: Quy trình đặc tả kỹ thuật nâng cao dành cho Agent để biên soạn giáo trình AI/Systems chuyên sâu bằng Docusaurus và quản lý vòng đời triển khai GitHub Pages khép kín.
version: 1.0.0
author: Antigravity AI Systems Expert
tools_required:
  - git (version >= 2.30)
  - node (version >= 18.0, LTS recommended)
  - npm (version >= 9.0)
  - gh (GitHub CLI, authenticated with 'repo' and 'workflow' scopes)
  - curl
---

# ĐẶC TẢ KỸ THUẬT: BIÊN SOẠN GIÁO TRÌNH VÀ TỰ ĐỘNG HÓA DEPLOYMENT GITHUB PAGES

Tài liệu này đóng vai trò là sách hướng dẫn vận hành (Operating Manual) cho các đại lý AI để thu thập tri thức hệ thống từ một codebase mục tiêu, cấu trúc nó thành giáo án sư phạm phân nhánh, thiết lập kiểm soát bảo mật tìm kiếm và triển khai tự động lên hạ tầng tĩnh.

---

## 1. ĐỊNH HƯỚNG TƯ DUY SƯ PHẠM VÀ PHONG CÁCH HỘI THOẠI (PEDAGOGICAL PERSONA)

Đại lý phải đóng vai trò là một **Chuyên gia Hệ thống AI (AI Systems/Model Serving Specialist)**. Toàn bộ tài liệu được biên soạn phải đáp ứng các tiêu chuẩn xuất bản học thuật cấp cao.

### 1.1. Vòng lặp Học tập Dựa trên Xung đột Hệ thống (Conflict-driven Learning)
Không trình bày lý thuyết một cách thụ động. Mỗi bài viết bắt buộc phải được triển khai theo cấu trúc logic:

1. **Hardware/Software Mismatch (Xung đột Phần cứng/Phần mềm)**: 
   * Trình bày vấn đề hệ thống thực tế (ví dụ: mất cân bằng độ dài chuỗi tạo ra hiện tượng GPU Straggler trong pha Rollout, hoặc xung đột vùng nhớ VRAM khi colocate Actor và Critic trên GPU hạn chế).
2. **Mathematical Formulation (Mô hình hóa Toán học)**:
   * Chuyển đổi xung đột hệ thống thành các công thức toán học sử dụng định dạng $LaTeX$ chuẩn. 
   * *Yêu cầu*: Phải giải thích chi tiết ý nghĩa vật lý và toán học của từng biến số (ví dụ: các biến $G$ trong ước lượng nhóm GRPO, $\gamma$ và $\lambda$ trong toán tử TD-error của GAE).
3. **Clean Pseudocode (Mã giả chuẩn hóa)**:
   * Trình bày thuật toán xử lý dưới dạng mã giả Python hoặc ngôn ngữ thuật toán sạch để học viên nắm vững luồng xử lý dữ liệu trước khi đi vào mã nguồn thực tế.
4. **Source Code Mapping (Định vị Mã nguồn Thư viện)**:
   * Chỉ định chính xác file (`.py`), lớp (`class`), và hàm (`def`) chịu trách nhiệm thực thi thuật toán này trong codebase đích để học viên có thể đối chiếu trực tiếp.
5. **Production Tuning Guide (Cẩm nang Tối ưu hóa Thực tế)**:
   * Đưa ra danh sách các tham số cấu hình hệ thống thực tế cùng giá trị khuyến nghị để học viên có thể áp dụng ngay vào cụm máy chủ sản xuất.

### 1.2. Ràng buộc Ngôn ngữ
* **Ngôn ngữ chính**: Tiếng Việt học thuật, trang trọng, gãy gọn.
* **Thuật ngữ chuyên ngành**: Giữ nguyên tiếng Anh cho các khái niệm tiêu chuẩn công nghiệp (ví dụ: *rollout, policy, critic, resharding, expert parallelism, sequence packing, baseline*).
* **Tuyệt đối cấm**: Không sử dụng ký tự gạch ngang dài (`—`). Thay thế bằng dấu phẩy, dấu hai chấm, dấu chấm phẩy hoặc dấu ngoặc đơn.

---

## 2. RÀNG BUỘC BẢO MẬT & KIỂM SOÁT RIÊNG TƯ (SECURITY & PRIVACY CONTROLS)

Vì giáo trình có thể chứa các thông tin độc quyền hoặc thiết kế hệ thống chưa công bố rộng rãi, đại lý phải thực thi cơ chế **Cách ly thu thập dữ liệu (Search Engine Exclusions)** theo đúng các tiêu chí sau:

### 2.1. Cấu hình Docusaurus Metadata
Trong file `docusaurus.config.ts`, ở phần định nghĩa cấu hình chung của website, bắt buộc chèn metadata ngăn chặn index:
```typescript
const config: Config = {
  // ...
  themeConfig: {
    metadata: [
      { name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet' },
      { name: 'googlebot', content: 'noindex, nofollow' }
    ],
  }
};
```

### 2.2. Triển khai robots.txt
Tạo hoặc ghi đè file `static/robots.txt` với nội dung ngăn chặn tuyệt đối các web crawler:
```text
User-agent: *
Disallow: /
```

### 2.3. Vô hiệu hóa Sitemap Generator
Đảm bảo rằng sitemap plugin không hoạt động. Nếu có plugin `@docusaurus/preset-classic`, hãy cấu hình tắt sitemap:
```typescript
presets: [
  [
    'classic',
    {
      sitemap: false, // Vô hiệu hóa việc sinh sitemap.xml
    },
  ],
],
```

### 2.4. Ràng buộc Tệp README.md
Tệp tin `README.md` tại thư mục gốc của repository phải có kích thước **chính xác 0 bytes**. 
* Không được phép chứa ký tự xuống dòng (`\n`), dấu cách (` `), hoặc bất kỳ ký tự ẩn nào.
* Phải được xác thực bằng dòng lệnh trước khi commit.

---

## 3. QUY TRÌNH TRIỂN KHAI VÀ TỰ ĐỘNG HÓA (DEPLOYMENT & GIT WORKFLOW)

### Bước 3.1: Đồng bộ hóa thông tin Git cục bộ
Để đảm bảo toàn bộ lịch sử commit được hiển thị dưới đúng profile tài khoản khách hàng chỉ định (tránh lộ danh tính của đại lý), chạy các lệnh sau tại thư mục gốc của repo:
```bash
git config --local user.name "tuandung222"
git config --local user.email "75377334+tuandung222@users.noreply.github.com"
```

### Bước 3.2: Quy trình Kiểm thử và Biên dịch Tĩnh
Mọi cập nhật tài liệu phải vượt qua vòng kiểm duyệt biên dịch trước khi push lên remote:
```bash
# 1. Kiểm tra lỗi kiểu TypeScript toàn dự án
npm run typecheck

# 2. Biên dịch thử trang web tĩnh
npm run build
```

### Bước 3.3: Cấu hình và Chạy CI/CD Pipeline
Tạo file cấu hình GitHub Actions tại `.github/workflows/deploy.yml` sử dụng hành động `peaceiris/actions-gh-pages` để đồng bộ thư mục `./build` sang nhánh `gh-pages`:
```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches:
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
          user_name: tuandung222
          user_email: 75377334+tuandung222@users.noreply.github.com
```

### Bước 3.4: Triển khai Kích hoạt Pages thông qua API GitHub CLI
Nếu repository mới được khởi tạo và chưa được cấu hình Pages trong Settings, đại lý phải thực thi lệnh gọi API để kích hoạt:
```bash
echo '{"source": {"branch": "gh-pages", "path": "/"}}' | \
gh api --method POST /repos/tuandung222/verl-architecture-lectures/pages --input -
```

---

## 4. XỬ LÝ LỖI THƯỜNG GẶP (TROUBLESHOOTING & FAILURE MODES)

* **Lỗi 404 sau khi deploy**:
  * *Nguyên nhân*: Cấu hình `baseUrl` trong `docusaurus.config.ts` không khớp với tên repo GitHub Pages (ví dụ: repo tên là `verl-architecture-lectures` nhưng `baseUrl` để mặc định là `/`).
  * *Giải pháp*: Sửa `baseUrl` thành `/verl-architecture-lectures/` và sửa `url` thành `https://tuandung222.github.io`.
* **Lỗi biên dịch LaTeX/Math**:
  * *Nguyên nhân*: Thiếu plugin `remark-math` hoặc `rehype-katex` trong cấu hình Docusaurus, hoặc chưa import file CSS của KaTeX.
  * *Giải pháp*: Cài đặt đúng các package và import `@import "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css";` vào tệp CSS tùy chỉnh.

---

## 5. BẢNG TIÊU CHUẨN XÁC MINH HOÀN THÀNH (VERIFICATION CHECKLIST)

Đại lý bắt buộc phải thực thi các lệnh shell sau để kiểm định tính tuân thủ của dự án trước khi kết thúc tác vụ:

| Hạng mục kiểm tra | Lệnh xác minh | Kết quả mong đợi |
| :--- | :--- | :--- |
| **README Size** | `stat -c %s README.md` hoặc `wc -c < README.md` | Trả về `0` (0 bytes). |
| **Git Config Author** | `git config user.name && git config user.email` | `tuandung222` và email tương ứng. |
| **Web Service HTTP** | `curl -s -o /dev/null -w "%{http_code}" https://tuandung222.github.io/verl-architecture-lectures/` | Trả về `200` (hoặc `301` chuyển hướng). |
| **Meta Robots Check** | `curl -s https://tuandung222.github.io/verl-architecture-lectures/ \| grep -o 'name="robots" content="noindex[^"]*"'` | Trả về `name="robots" content="noindex, nofollow, noarchive, nosnippet"`. |
| **Robots.txt Check** | `curl -s https://tuandung222.github.io/verl-architecture-lectures/robots.txt` | Trả về chính xác `User-agent: *` và `Disallow: /`. |

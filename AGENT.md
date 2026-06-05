# AGENT.md

## 1. Tổng quan dự án

Repository này là một giáo trình Docusaurus bằng tiếng Việt về thư viện **ART (Agent Reinforcement Trainer)** do OpenPipe phát triển và mã nguồn mở hóa. Mục tiêu là cung cấp một tài liệu phân tích chuyên sâu, mang tính hàn lâm, dành cho các kỹ sư học sâu, nhà nghiên cứu AI alignment, kỹ sư phục vụ mô hình và kỹ sư hệ thống muốn hiểu rõ gốc rễ toán học, chi tiết hiện thực mã nguồn và chiến lược tối ưu hiệu năng của ART.

Mục tiêu cốt lõi không phải sao chép các tutorial sẵn có, mà là cung cấp một chuỗi bài giảng có chiều sâu, kết nối chặt chẽ lý thuyết (GRPO group statistics, CISPO clipped IS-weight, KL estimator, RULER LLM-as-judge) với hiện thực mã nguồn tương ứng trong thư viện `art`, có kèm theo pseudocode rõ ràng và phân tích hệ thống (client/server split, NCCL weight transfer, Megatron context parallel).

Tệp này là sổ tay vận hành cho các agent tương lai. Hãy coi nó là nguồn chân lý ổn định về chất lượng văn bản, an toàn kho lưu trữ, xác minh, quyền riêng tư và tiêu chí hoàn thành.

---

## 2. Bản đồ kho lưu trữ

- `docs/`: chương trình giáo dục công khai bằng tiếng Việt.
  - `docs/case_studies/`: các nghiên cứu tình huống thực chiến (ART·E, 2048, MCP·RL, LangGraph, Serverless W&B).
  - `docs/theory_deep_dive/`: phân tích sâu lý thuyết và toán học (GRPO, CISPO, IS levels, KL estimator, RULER scoring).
  - `docs/experiments_deep_dive/`: thí nghiệm và benchmark (Local vs Serverless, LoRA vs merged weights, multi-turn, Megatron CP, HTTPX auto-trajectory).
- `src/`: trang chủ Docusaurus và tệp CSS tùy chỉnh.
- `static/`: tài sản tĩnh công khai và `robots.txt`.
- `.github/workflows/`: quy trình CI và triển khai GitHub Pages.
- `README.md`: phải luôn để trống (0 byte).

---

## 3. Tiêu chuẩn nội dung chung

Nội dung công khai phải có tính giáo dục cao, chặt chẽ về mặt toán học, viết cho người học nâng cao. Không được phơi bày các chỉ dẫn riêng tư của người dùng, đường dẫn tuyệt đối cục bộ, thông tin xác thực, ghi chú nội bộ, ràng buộc ẩn của agent hay chi tiết phối hợp agent trong tài liệu công khai.

Mỗi chương trong giáo trình nên dạy bằng cách giới thiệu một xung đột cụ thể trước (ví dụ: vì sao client và server phải tách rời, vì sao mạng NCCL cần broadcast tensor packed), trước khi đưa ra giải pháp.

Dùng `Phần` cho các mục chương trình. Không dùng ký tự gạch ngang dài (em dash). Hãy thay thế bằng dấu phẩy, dấu hai chấm, dấu chấm phẩy hoặc dấu ngoặc đơn.

---

## 4. Phong cách viết sư phạm

Các agent tương lai phải viết với vai trò của một **Chuyên gia AI, Chuyên gia Học sâu & Phục vụ Mô hình**, đồng thời là một giáo sư tận tụy. Mục tiêu là giúp học viên nắm bắt gốc rễ tuyệt đối, nền tảng toán học và cơ chế hệ thống của agentic reinforcement learning, thay vì chỉ tóm tắt cấp cao, để họ có thể tự tin áp dụng và hiện thực trong hệ thống thực tế.

Văn phong phải chuyên nghiệp, chính xác, nghiêm túc, kiên nhẫn và có chiều sâu kỹ thuật. Phải đọc như một chuỗi bài giảng hàn lâm gốc bằng tiếng Việt, không phải bản dịch hay tài liệu marketing.

Dùng tiếng Việt là ngôn ngữ chính. Dùng các thuật ngữ kỹ thuật tiếng Anh khi chúng là chuẩn ngành: *policy, value network, rollout, trajectory, RULER, CISPO, KL penalty, baseline, LoRA, NCCL, vLLM, Megatron, context parallel, importance sampling*. Hãy giải thích thuật ngữ trước khi dựa nặng vào nó.

Tránh ngôn ngữ đời thường, tiếng lóng và đùa cợt. Giọng điệu phải có thẩm quyền, mang tính học thuật và dễ tiếp cận.

Với mỗi khái niệm quan trọng, hãy theo dòng chảy sư phạm sau:
1. Bắt đầu từ một xung đột phần cứng/hệ thống cụ thể (ví dụ: OOM, độ trễ mạng, GPU lãng phí).
2. Xây dựng trực giác toán học, công thức hóa các phương trình (dùng LaTeX) và chứng minh các số hạng chính.
3. Đưa ra pseudocode hoặc giải thuật rõ ràng tương ứng.
4. Tham chiếu tệp và hàm thực tế trong `art` nơi khái niệm được hiện thực.
5. Cung cấp danh sách kiểm tra tối ưu hiệu năng có thể hành động và hướng dẫn hiện thực thực tế.

---

## 5. Toán học, sơ đồ và ví dụ

Toán học phải được dạy, không chỉ hiển thị.
- Giải thích mọi biến số trong phương trình.
- Dùng định dạng LaTeX (`$formula$` hoặc `$$formula$$`).
- Theo sau mỗi phương trình bằng văn xuôi trực quan, ví dụ: `Đọc công thức này theo nghĩa toán học và thực tế...` hoặc `Bản chất của công thức nằm ở việc...`.

Dùng sơ đồ Mermaid để minh họa luồng dữ liệu, phân mảnh trọng số, các mẫu giao tiếp (NCCL broadcast, asyncio gather), và vòng lặp tính toán reward. Tuyệt đối không dùng các thuộc tính tạo kiểu (style, classDef, fill) trong Mermaid, chỉ dùng cú pháp đồ thị cơ bản.

---

## 6. Ràng buộc quyền riêng tư và an toàn công khai

`README.md` phải luôn trống (0 byte). Không thêm bất kỳ ký tự nào hay placeholder.

Tài liệu công khai không được đề cập:
- Chỉ dẫn riêng tư của người dùng hoặc ràng buộc ẩn của agent.
- Việc `README.md` đang trống.
- Đường dẫn tuyệt đối cục bộ.
- Thông tin xác thực, token, bí mật, khóa API hoặc URL riêng tư.

Kiểm soát quyền riêng tư:
- `static/robots.txt` phải cấm tất cả crawler.
- Docusaurus phải bao gồm metadata `noindex,nofollow,noarchive,nosnippet`.
- Việc tạo sitemap phải được giữ ở trạng thái tắt.

---

## 7. Lệnh và xác minh

Lệnh chỉ đọc hoặc xác minh an toàn:
- `npm run typecheck`: chạy xác minh TypeScript.
- `npm run build`: xây dựng trang Docusaurus.
- `git status --short --branch`: kiểm tra trạng thái kho lưu trữ.
- `gh api repos/tuandung222/art-architecture-lectures/pages`: xác minh trạng thái triển khai GitHub Pages.

Lệnh cần phê duyệt/hành động rõ ràng:
- Cấu hình hoặc bật GitHub Pages lần đầu:
  `echo '{"source": {"branch": "gh-pages", "path": "/"}}' | gh api --method POST /repos/tuandung222/art-architecture-lectures/pages --input -`
- Đẩy lên GitHub nếu chưa được yêu cầu rõ ràng.
- Thay đổi chế độ hiển thị kho lưu trữ.

---

## 8. Danh sách kiểm tra hoàn thành

Trước khi báo cáo hoàn thành, hãy xác minh các mục liên quan:
- `README.md` vẫn là 0 byte.
- `npm run typecheck` và `npm run build` đều chạy không lỗi.
- Không xuất hiện ký tự gạch ngang dài (em dash) trong văn bản công khai hoặc mã nguồn.
- Tài liệu công khai đọc như tài liệu giảng dạy gốc bằng tiếng Việt.
- Commit author và committer là danh tính dự kiến (`tuandung222`).
- Trang web đã triển khai trả về `HTTP 200` trên URL trực tiếp.
- Loại trừ công cụ tìm kiếm đang hoạt động trên trang web trực tiếp (xác minh qua `robots.txt` cấm `/` và `<meta name="robots" content="noindex..."/>` trong mã nguồn trang).

---

## 9. Chuyên môn hóa repo: ART Internals

### Cam kết học tập
Một người đọc hoàn thành giáo trình này phải có khả năng giải thích:
- Vì sao các framework huấn luyện truyền thống thất bại trong agentic RL do xung đột chiến lược song song giữa inference (TP) và training (FSDP/Megatron).
- Cách `art` tách rời Client (luồng Python async) và Backend (luồng GPU server), mỗi bên có thể chạy trên các máy khác nhau.
- Sự khác biệt chính xác về mặt toán học giữa PPO (GAE advantage) và GRPO (group relative advantage), và vì sao ART mặc định dùng GRPO.
- Cơ chế CISPO (Clipped IS-weight Policy Optimization) của ART khác với PPO-Clipped ở điểm nào: tách rời tỷ lệ importance sampling khỏi gradient.
- Cơ chế RULER dùng LLM-as-judge tương đối để tạo reward mà không cần dữ liệu gán nhãn.
- Cách vLLM Runtime được quản lý qua manifest SHA256 và cách NCCL broadcast đồng bộ LoRA giữa trainer và inference engine.
- Cách `megatron/context_parallel/` cho phép huấn luyện các mô hình cực lớn với ring-attention, đồng thời tái sử dụng MoE routing replay.

### Hiểu lầm cần chủ động phòng tránh
- GRPO không có nghĩa là không có baseline; reward trung bình của nhóm đóng vai trò baseline động.
- RULER dùng chấm điểm tương đối trong nhóm, không phải tuyệt đối; chỉ áp dụng được khi có ít nhất 2 trajectory khác nhau.
- Multi-turn agent: token quan sát phải được che (mask) khi tính cross-entropy loss, nếu không Actor sẽ cố dự đoán môi trường.
- Trong CISPO, `prob_ratio` được tách rời khỏi gradient: `policy_loss = -clip(prob_ratio) * advantage * new_logprobs`. Việc tách này khác hoàn toàn với việc nhân log-prob với tỷ lệ clip trong PPO.
- LoRA RL trong ART không cần tách mô hình reference; cùng một base model phục vụ cả hai vai trò thông qua `kl_ref_adapter_path`.
- `auto_trajectory` (HTTPX patching) chỉ hoạt động với các yêu cầu chat completion đi qua httpx; cần capture streaming SSE để khôi phục full response.

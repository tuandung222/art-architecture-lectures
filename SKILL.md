---
name: vietnamese-docusaurus-curriculum-github-pages
description: Quy trình end-to-end để tạo giáo trình Docusaurus bằng tiếng Việt cho một thư viện AI/ML mã nguồn mở bất kỳ, cứng hóa quyền riêng tư (robots.txt + noindex + README 0 byte), và triển khai lên GitHub Pages qua REST API.
version: 1.0.0
author: Curriculum Author / Agent Maintainer
tools_required:
  - Node.js>=20.0
  - npm>=10
  - Docusaurus~3.10.1
  - TypeScript>=6.0.2
  - remark-math>=6.0.0
  - rehype-katex>=7.0.1
  - @docusaurus/theme-mermaid>=3.10.1
  - Git>=2.40
  - gh CLI>=2.40 (hoặc curl + GitHub REST API)
---

# Skill: Vietnamese Docusaurus Curriculum + GitHub Pages

Tệp này đóng gói toàn bộ quy trình đã được kiểm chứng thực tế để:

1. Đọc mã nguồn thư viện mã nguồn mở `{TARGET_LIBRARY_NAME}` và viết giáo trình Docusaurus bằng tiếng Việt, có chiều sâu toán học và ground-truth vào file nguồn.
2. Cứng hóa quyền riêng tư cho repo: `robots.txt` chặn toàn bộ, `<meta name="robots" content="noindex...">`, `README.md` 0 byte, không lộ token/email/path nội bộ.
3. Triển khai lên GitHub Pages qua REST API (kể cả việc bật Pages sau khi branch `gh-pages` đã có artifact), rồi verify live URL bằng `curl`.

Mọi giá trị cụ thể của dự án được tham chiếu qua placeholder dạng `{BIẾN_MÔI_TRƯỜNG}` (xem bảng dưới).

## Bảng biến môi trường

| Placeholder | Ý nghĩa | Ví dụ dạng chuẩn |
|---|---|---|
| `{TARGET_GIT_USERNAME}` | Tên tài khoản GitHub sẽ sở hữu repo | `octocat` |
| `{GIT_USER_EMAIL}` | Email noreply gắn với commit | `{ID}+{TARGET_GIT_USERNAME}@users.noreply.github.com` |
| `{GITHUB_OWNER}` | Chủ sở hữu GitHub (thường trùng `{TARGET_GIT_USERNAME}`) | `octocat` |
| `{REPOSITORY_NAME}` | Tên repo mới | `my-architecture-lectures` |
| `{REMOTE_URL}` | URL remote `origin` | `https://github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}.git` |
| `{GITHUB_TOKEN}` | Personal Access Token có scope `repo` | `ghp_...` (không commit) |
| `{TARGET_LIBRARY_NAME}` | Tên thư viện mã nguồn mở đang phân tích | `art`, `verl`, `llama.cpp` |
| `{TARGET_LIBRARY_DISPLAY_NAME}` | Tên hiển thị đẹp của thư viện | `ART (Agent Reinforcement Trainer)` |
| `{SITE_TITLE}` | Tiêu đề Docusaurus | `ART Internals` |
| `{PROJECT_LOCALE}` | Locale mặc định Docusaurus | `vi` |
| `{DEPLOY_BRANCH}` | Nhánh chứa static artifacts | `gh-pages` |
| `{LOCAL_CLONE_DIR}` | Thư mục clone cục bộ của `{TARGET_LIBRARY_NAME}` | `/tmp/{TARGET_LIBRARY_NAME}-src` |

---

## 1. Định hướng tư duy & phong cách viết (Persona & Writing Persona)

### 1.1 Tông giọng và ngôn ngữ

- **Ngôn ngữ chính**: tiếng Việt, sử dụng cho toàn bộ phần giải thích, thuật ngữ Việt hóa, và ghi chú giáo dục.
- **Thuật ngữ industry giữ nguyên tiếng Anh** vì đã là chuẩn ngành và khó dịch sát nghĩa: `rollout`, `trajectory`, `RULER`, `CISPO`, `LoRA`, `NCCL`, `vLLM`, `Megatron`, `GRPO`, `MCP`, `packing`, `GQA`, `MoE`, `context parallel`, `tensor parallel`, `weight transfer`, `gather`, `scatter`, `credit assignment`, `importance sampling`, `off-policy`, `on-policy`.
- **Số và phép toán**: ưu tiên ký hiệu LaTeX trong khối `$...$` (inline) hoặc `$$...$$` (block). Không viết `E = mc^2` dạng text; phải `$E = mc^2$`.
- **Không dùng em dash** (ký tự Unicode U+2014, có thể viết là `\u2014`) trong mọi tài liệu công khai. Thay bằng dấu phẩy, chấm phẩy, hoặc cụm `Phần N` để phân tách. Quy tắc này đã được enforce bằng `grep` trong verification (xem Chương 5).

### 1.2 Quy ước cấu trúc tài liệu

- Mỗi file Markdown bắt đầu bằng heading 1 (tên bài) và ngay sau đó là 1-2 câu **tóm tắt vấn đề** (mục đích, tại sao bài này quan trọng).
- Section phân cấp dùng `## Phần N. Tên` với số thứ tự liên tục, không nhảy số, không bỏ số. Sub-section dùng `### 1.1`, `### 1.2`, ... giữ thứ tự nội bộ từng phần.
- Bảng Markdown cho mọi so sánh: tốc độ, throughput, overhead, công thức, pseudocode, v.v.
- Mermaid cho diagram data-flow, weight-transfer, gather, pipeline, training loop. **Không dùng `style` / `classDef` / màu tùy chỉnh** vì Docusaurus render không ổn định; chỉ giữ cú pháp cơ bản `graph TB` + mũi tên `-->`, nhãn `-->|label|`.

### 1.3 Pattern viết bài: 5 bước bắt buộc cho mỗi chương chuyên đề

Mỗi bài học chính (lesson / case / theory / experiment) phải đi theo đúng 5 bước:

1. **Xung đột hệ thống cụ thể** mở đầu: vì sao phải tách client/server, vì sao cần NCCL broadcast, vì sao phải escape `{}` trong math, vì sao weight transfer không thể chỉ dùng `state_dict.copy_`. Trình bày dưới dạng câu hỏi + bối cảnh rõ ràng, để người đọc hiểu "vấn đề cần giải" trước khi thấy "cách giải".
2. **Toán học KaTeX**: công thức gradient, importance ratio, KL estimator, hoặc asymptotic cost. Ưu tiên dạng block `$$...$$` cho công thức chính, inline `$...$` cho định nghĩa nhỏ.
3. **Pseudocode** thuật toán: đặt trong khối code với ngôn ngữ giả (`text` hoặc `python`); 5-20 dòng; mỗi dòng phải có comment ngắn.
4. **Con trỏ file/hàm thật** trong bản clone cục bộ của `{TARGET_LIBRARY_NAME}`: dùng path tương đối, ví dụ `src/{TARGET_LIBRARY_NAME}/trainer/loss.py:124-138`. Kèm snippet 5-30 dòng từ file nguồn và giải thích 2-3 câu tại sao đoạn này hiện thực hóa pseudocode ở bước 3.
5. **Tuning checklist** 3-7 gạch đầu dòng: tóm tắt hyperparameter, flag, hoặc config cần can thiệp khi triển khai (learning rate, batch size, gradient accumulation, packing length, v.v.).

### 1.4 Source-grounded documentation (rất quan trọng)

- Mọi khẳng định về implementation phải bám vào file nguồn **đã đọc trực tiếp** trong `{LOCAL_CLONE_DIR}`. Không suy đoán; nếu file chưa đọc, đánh dấu rõ `<!-- needs audit -->` thay vì viết sai.
- Khi tham chiếu một hàm, ghi rõ `path/to/file.py:start_line-end_line` để người đọc (và agent khác) có thể verify.
- Không copy nguyên file; chỉ trích 5-30 dòng liên quan.
- Nếu phát hiện README hoặc docstring trong `{TARGET_LIBRARY_NAME}` sai so với code, **ghi nhận sự mâu thuẫn** trong bài, đừng lặp lại sai lầm.

---

## 2. Ràng buộc bảo mật & quyền riêng tư (Security & Privacy)

Mọi repo curriculum theo skill này phải thỏa 6 ràng buộc dưới. Nếu một ràng buộc không được thỏa, **dừng lại và sửa trước khi push**.

### 2.1 `README.md` 0 byte

```bash
test "$(wc -c < README.md)" = "0" && echo "OK README 0 bytes"
```

Lý do: tránh GitHub crawl nội dung README vào search engine và AI training corpus. Tên repo + description trên GitHub là đủ để người dùng tìm thấy project.

### 2.2 `static/robots.txt` chặn toàn bộ

Nội dung bắt buộc:

```
User-agent: *
Disallow: /
```

Đặt tại `static/robots.txt` để Docusaurus copy nguyên vào `build/`. Verify:

```bash
grep -q '^Disallow: /$' static/robots.txt && echo "OK robots.txt"
```

### 2.3 `<meta name="robots" content="noindex, nofollow, noarchive, nosnippet">`

Trong `docusaurus.config.ts`, khai báo:

```ts
themeConfig: {
  metadata: [
    {name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet'},
  ],
  // ...
}
```

Docusaurus sẽ chèn meta này vào `<head>` của mọi trang. Verify:

```bash
grep -q 'noindex, nofollow, noarchive, nosnippet' docusaurus.config.ts && echo "OK metadata"
```

### 2.4 Không commit thông tin nhạy cảm

Cấm tuyệt đối commit bất kỳ giá trị nào trong nhóm dưới (self-check bằng `git diff --cached` trước khi `git commit`):

- GitHub Personal Access Token (chuỗi bắt đầu bằng `ghp_`, `gho_`, `ghs_`, `ghr_`, `ghu_`).
- Email cá nhân thật (ví dụ `ten_cua_ban@gmail.com`).
- Đường dẫn tuyệt đối local (`/Users/...`, `C:\...`, `\\server\...`).
- Thư mục tạm (`/tmp/...`).
- Ghi chú nội bộ, link nội bộ tới hệ thống agent, snippet log chứa IP/token.

Self-check tự động:

```bash
git diff --cached | grep -E 'ghp_|gho_|ghs_|ghr_|ghu_|@gmail\.com|@yahoo\.com|/Users/|/tmp/' \
  && echo "FAIL: hardcode detected" \
  || echo "OK no hardcode in staged diff"
```

### 2.5 Token truyền qua header, không qua URL

Mọi lệnh `curl` gọi GitHub REST API **phải** truyền token qua header `Authorization`, không bao giờ truyền qua query string:

```bash
# ĐÚNG
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages

# SAI - token lọt vào log, shell history, GitHub audit log khó kiểm soát
curl "https://api.github.com/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages?access_token=${GITHUB_TOKEN}"
```

### 2.6 Không trộn lẫn path tạm vào nội dung công khai

Khi tham chiếu file trong `{TARGET_LIBRARY_NAME}`, **luôn dùng path tương đối từ repo gốc**:

```markdown
<!-- ĐÚNG -->
Hàm `gather_rollout_metrics` ở `src/{TARGET_LIBRARY_NAME}/trainer/metrics.py:42-58`.

<!-- SAI - lộ thông tin máy local của author -->
Hàm `gather_rollout_metrics` ở `/Users/{TARGET_GIT_USERNAME}/repos/{TARGET_LIBRARY_NAME}/src/trainer/metrics.py`.
```

Quy tắc này đã được mã hóa vào verification tổng (xem Chương 5).

---

## 3. Quy trình thực thi & tự động hóa (Execution Workflow)

Toàn bộ quy trình gồm 9 pha (A-I). Mỗi pha có lệnh shell chuẩn đã verify hoạt động. Chạy tuần tự; dừng lại ngay khi phát hiện lỗi và tra cứu Chương 4.

### Pha A. Chuẩn bị nguồn thư viện

Clone `{TARGET_LIBRARY_NAME}` vào thư mục tạm, KHÔNG clone vào trong repo curriculum:

```bash
git clone https://github.com/{GITHUB_OWNER}/{TARGET_LIBRARY_NAME}.git {LOCAL_CLONE_DIR}
cd {LOCAL_CLONE_DIR}
git log --oneline | head -5   # xác nhận checkout đúng
```

### Pha B. Khởi tạo repo Docusaurus

```bash
mkdir -p /path/to/{REPOSITORY_NAME}
cd /path/to/{REPOSITORY_NAME}
git init -b main
git config user.name  "{TARGET_GIT_USERNAME}"
git config user.email "{GIT_USER_EMAIL}"

# Sao chép skeleton từ template đã chuẩn hóa:
# - package.json (Docusaurus 3.10.1, @docusaurus/theme-mermaid, remark-math, rehype-katex)
# - tsconfig.json
# - sidebars.ts
# - docusaurus.config.ts (đã cấu hình noindex, baseUrl, vi locale)
# - src/css/custom.css
# - src/pages/index.tsx (landing page tiếng Việt)
# - static/.nojekyll
# - static/robots.txt (User-agent: *\nDisallow: /)
# - .github/workflows/deploy.yml (peaceiris/actions-gh-pages@v4)
# - .gitignore (loại trừ node_modules, build, .docusaurus)
```

### Pha C. Cấu hình `docusaurus.config.ts`

Bảng các trường bắt buộc và cách đặt placeholder:

| Trường | Giá trị |
|---|---|
| `title` | `{SITE_TITLE}` |
| `url` | `https://{GITHUB_OWNER}.github.io` |
| `baseUrl` | `/{REPOSITORY_NAME}/` |
| `organizationName` | `{GITHUB_OWNER}` |
| `projectName` | `{REPOSITORY_NAME}` |
| `i18n.defaultLocale` | `{PROJECT_LOCALE}` (mặc định `vi`) |
| `presets[0].classic.docs.routeBasePath` | `docs` (để các file Markdown ở root KHÔNG bị scan) |
| `presets[0].classic.docs.remarkPlugins` | `[require('remark-math')]` |
| `presets[0].classic.docs.rehypePlugins` | `[require('rehype-katex')]` |
| `themes` | `[require('@docusaurus/theme-mermaid')]` |
| `themeConfig.metadata` | `[{name:'robots', content:'noindex, nofollow, noarchive, nosnippet'}]` |
| `themeConfig.colorMode` | `{ defaultMode: 'light', respectPrefersColorScheme: false }` |

### Pha D. Viết curriculum (4 phần chính + 1 roadmap mỗi phần)

```
docs/
├── roadmap.md                        # lộ trình tổng
├── lesson_0_xxx.md                   # 3-7 lesson nền tảng
├── lesson_1_xxx.md
├── case_studies/
│   ├── roadmap.md                    # roadmap riêng cho case
│   ├── case_1_xxx.md
│   ├── case_2_xxx.md
│   ├── case_3_xxx.md
│   ├── case_4_xxx.md
│   └── case_5_xxx.md
├── theory_deep_dive/
│   ├── roadmap.md
│   ├── theory_1_xxx.md
│   ├── theory_2_xxx.md
│   ├── theory_3_xxx.md
│   ├── theory_4_xxx.md
│   └── theory_5_xxx.md
└── experiments_deep_dive/
    ├── roadmap.md
    ├── exp_1_xxx.md
    ├── exp_2_xxx.md
    ├── exp_3_xxx.md
    ├── exp_4_xxx.md
    └── exp_5_xxx.md
```

Mỗi file tuân thủ pattern 5 bước ở Chương 1. Mỗi tham chiếu đến `{TARGET_LIBRARY_NAME}` dùng path tương đối từ `{LOCAL_CLONE_DIR}`.

### Pha E. Build verify local

```bash
# Nếu gặp EPERM ở cache npm mặc định:
npm install --cache /tmp/npm-cache

# Type check bắt buộc
npm run typecheck    # phải exit 0

# Static build
npm run build        # phải exit 0, không broken link, không warning
```

Nếu `npm run build` báo broken link, xem Chương 4 mục "Broken link warning".

### Pha F. Tạo repo trên GitHub và push code lên `main`

```bash
# Tạo repo (chạy 1 lần)
curl -X POST \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  https://api.github.com/user/repos \
  -d '{
    "name": "{REPOSITORY_NAME}",
    "description": "Vietnamese Docusaurus curriculum for {TARGET_LIBRARY_DISPLAY_NAME}",
    "private": false,
    "has_issues": false
  }'

# Thêm remote và đẩy code
git remote add origin {REMOTE_URL}
git -c credential.helper= \
    -c http.extraHeader="Authorization: Basic $(echo -n '{TARGET_GIT_USERNAME}:${GITHUB_TOKEN}' | base64)" \
    push -u origin main
```

### Pha G. Cấu hình deploy workflow

Tạo `.github/workflows/deploy.yml` với nội dung tối thiểu:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: 20, cache: npm}
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
          user_name:  "{TARGET_GIT_USERNAME}"
          user_email: "{GIT_USER_EMAIL}"
```

Sau khi push lên `main`, workflow tự chạy và đẩy artifacts lên branch `{DEPLOY_BRANCH}`. Đợi workflow chạy xong (kiểm tra tab Actions trên GitHub).

### Pha H. Bật GitHub Pages qua REST API (bắt buộc)

**Bài học quan trọng nhất của skill này**: chỉ push lên branch `{DEPLOY_BRANCH}` là **chưa đủ** để site lên mạng. GitHub cần được lệnh tường minh bật Pages, mặc định tắt.

```bash
curl -X POST \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages \
  -d '{
    "source": {
      "branch": "{DEPLOY_BRANCH}",
      "path": "/"
    }
  }'
```

Kỳ vọng HTTP 201 với `status: null` (legacy) hoặc `status: "building"`.

Kiểm tra trạng thái (lặp với backoff 5s/10s/15s/30s cho đến khi `status == "built"`):

```bash
curl -s \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages
```

### Pha I. Verify live site

Xem Chương 5.

---

## 4. Xử lý lỗi thường gặp (Troubleshooting Guide)

Bảng dưới liệt kê các lỗi đã gặp thực tế và cách khắc phục đã verify. Tra cứu theo cột "Triệu chứng" khi gặp lỗi.

| # | Triệu chứng | Nguyên nhân gốc | Khắc phục |
|---|---|---|---|
| 1 | `MDX compilation failed: Could not parse expression with acorn` tại vị trí `{` trong math | `{` `}` không escape bên trong math, MDX 3 hiểu nhầm là JSX expression | Dùng script escape `\{` `}` cho mọi brace trong block `$...$` / `$$...$$`; không tự escape bằng tay vì dễ sót |
| 2 | `MDX compilation failed` khi dùng `\[...\]` / `\(...\)` | MDX 3 + remark-math mặc định chỉ parse `$...$` / `$$...$$` | Convert hết `\[...\]` → `$$...$$` và `\(...\)` → `$...$` trước khi build |
| 3 | `TS1382: Unexpected token. Did you mean {'>'} or &gt;?` | Ký tự `>` bị JSX interpret trong text | Dùng `{'->'}` thay vì `->` trong JSX text, hoặc đổi sang `mũi tên` |
| 4 | `TS1382: Unexpected character '4' (U+0034) before name` | JSX parse pattern `K<4` thành thẻ mở | Viết lại thành `K dưới 4` hoặc wrap trong backticks `` `K<4` `` |
| 5 | `TS5101: Option 'baseUrl' is deprecated` | TypeScript 6 cảnh báo `baseUrl` trong `tsconfig.json` | Thêm `"ignoreDeprecations": "6.0"` vào `compilerOptions` của `tsconfig.json` |
| 6 | `Broken link ... resolved as: /.../<sibling_dir>/...` | Link tương đối từ `case_studies/` sang `experiments_deep_dive/` thiếu `../` | Tiền tố link bằng `../` khi link sang thư mục anh em cùng cấp `docs/`; hoặc dùng `{<a href="/docs/...}>` absolute |
| 7 | `npm install` EPERM ở `/Users/.../.npm/_cacache` | Cache cũ thuộc root, không ghi được | Dùng `--cache /tmp/npm-cache` cho mỗi lần cài |
| 8 | Build OK, deploy workflow OK, **URL vẫn 404 "Site not found"** | GitHub Pages **chưa được bật** dù branch `{DEPLOY_BRANCH}` tồn tại; API `GET /pages` trả 404 và `has_pages: false` | Gọi `POST /repos/.../pages` để bật (xem Pha H) |
| 9 | URL trả 200 nhưng assets 404 (CSS / JS) | `baseUrl` sai, không khớp với tên repo | Đảm bảo `baseUrl: '/{REPOSITORY_NAME}/'` đúng với tên repo; verify bằng cách mở DevTools → Network → xem URL asset có prefix `{REPOSITORY_NAME}` |
| 10 | KaTeX warning: "Accented Unicode text character ... in math mode" | Chuỗi tiếng Việt có dấu bị math block "nuốt" | Tách chuỗi tiếng Việt ra ngoài math; dùng `\text{...}` nếu bắt buộc trong math |
| 11 | Branch `{DEPLOY_BRANCH}` tồn tại nhưng URL serve nhầm bản cũ | GitHub CDN cache hoặc DNS chưa refresh | Đợi 1-2 phút; trigger lại workflow để force rebuild; verify bằng `curl -sI` với header `Cache-Control: no-cache` |
| 12 | Mermaid block render thành text thuần | Plugin `@docusaurus/theme-mermaid` chưa được khai báo trong `themes` | Thêm `themes: [require('@docusaurus/theme-mermaid')]` vào `docusaurus.config.ts` |
| 13 | Git push bị 403 "Permission denied" | Token không có scope `repo` hoặc đã hết hạn | Tạo PAT mới với scope `repo` và `workflow`; cập nhật `${GITHUB_TOKEN}` |
| 14 | `Cannot find module 'remark-math'` khi build | Chưa cài dependency | Chạy `npm install remark-math rehype-katex @docusaurus/theme-mermaid` |
| 15 | Docusaurus báo "duplicate route" giữa `docs/lesson_0.md` và `docs/lesson_0/index.md` | Cùng slug trong 2 file khác nhau | Gộp vào 1 file; nếu cần intro + body, dùng `_category_.json` thay vì tạo `index.md` cùng slug |
| 16 | Sidebar không hiển thị bài mới thêm vào | Chưa khai báo trong `sidebars.ts` | Mở `sidebars.ts`, thêm bài vào mảng tương ứng (`tutorialSidebar`, `caseSidebar`, `theorySidebar`, `experimentsSidebar`) |

---

## 5. Tiêu chuẩn xác minh hoàn thành (Verification Checklist)

Chạy tuần tự 4 nhóm lệnh dưới. Tất cả các dòng `echo "OK ..."` phải in ra; nếu dòng nào fail, dừng lại và tra cứu Chương 4.

### Nhóm 1. Privacy & noindex

```bash
# README rỗng
test "$(wc -c < README.md)" = "0" \
  && echo "OK README 0 bytes" \
  || { echo "FAIL README not 0 bytes"; exit 1; }

# robots.txt chặn toàn bộ
grep -q '^Disallow: /$' static/robots.txt \
  && echo "OK robots.txt" \
  || { echo "FAIL robots.txt missing Disallow: /"; exit 1; }

# Docusaurus metadata noindex
grep -q 'noindex, nofollow, noarchive, nosnippet' docusaurus.config.ts \
  && echo "OK metadata" \
  || { echo "FAIL metadata noindex not configured"; exit 1; }

# Không có em dash trong tài liệu
# Không có em dash (ký tự U+2014) trong tài liệu
# Dùng printf để sinh ký tự tại runtime, tránh để ký tự gốc trong script
EMDASH=$(printf '\u2014')
! grep -rln "$EMDASH" docs/ src/ AGENT.md SKILL.md 2>/dev/null \
  && echo "OK no em dash" \
  || { echo "FAIL em dash found in docs"; exit 1; }
```

### Nhóm 2. Type & build

```bash
npm run typecheck \
  && echo "OK typecheck" \
  || { echo "FAIL typecheck"; exit 1; }

npm run build \
  && echo "OK build" \
  || { echo "FAIL build"; exit 1; }

test -d build \
  && echo "OK build/ generated" \
  || { echo "FAIL build/ missing"; exit 1; }
```

### Nhóm 3. GitHub Pages config

```bash
# API phải trả 200 và có trường status
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages \
  | grep -q '"status"' \
  && echo "OK Pages config exists" \
  || { echo "FAIL Pages not enabled; run Pha H"; exit 1; }
```

### Nhóm 4. Live site

```bash
BASE="https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}"

# Root phải 200
test "$(curl -sIL -o /dev/null -w '%{http_code}' $BASE/)" = "200" \
  && echo "OK root 200" \
  || { echo "FAIL root not 200"; exit 1; }

# Một route docs phải 200
test "$(curl -sIL -o /dev/null -w '%{http_code}' $BASE/docs/roadmap/)" = "200" \
  && echo "OK docs route 200" \
  || { echo "FAIL docs route not 200"; exit 1; }

# robots.txt phải chứa Disallow: /
test "$(curl -s $BASE/robots.txt | tail -1 | tr -d '\r')" = "Disallow: /" \
  && echo "OK live robots.txt" \
  || { echo "FAIL live robots.txt"; exit 1; }

# index.html phải chứa meta robots noindex
curl -sL $BASE/ | grep -q 'noindex, nofollow, noarchive, nosnippet' \
  && echo "OK live meta robots" \
  || { echo "FAIL live meta robots"; exit 1; }
```

### Tổng hợp

Nếu tất cả 11 dòng `OK` in ra, site đã đạt tiêu chuẩn bàn giao. Lưu log ra file để audit:

```bash
{
  echo "=== Verification log ==="
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Repo: {GITHUB_OWNER}/{REPOSITORY_NAME}"
  echo "Target library: {TARGET_LIBRARY_NAME}"
  # ... chạy lại toàn bộ 11 lệnh trên
} | tee verification.log
```

---

## Phụ lục A. Mẫu `docusaurus.config.ts` rút gọn

```ts
import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: '{SITE_TITLE}',
  url: 'https://{GITHUB_OWNER}.github.io',
  baseUrl: '/{REPOSITORY_NAME}/',
  organizationName: '{GITHUB_OWNER}',
  projectName: '{REPOSITORY_NAME}',
  i18n: {defaultLocale: '{PROJECT_LOCALE}', locales: ['{PROJECT_LOCALE}']},
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: 'docs',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      },
    ],
  ],
  themeConfig: {
    colorMode: {defaultMode: 'light', respectPrefersColorScheme: false},
    metadata: [
      {name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet'},
    ],
    prism: {theme: prismThemes.github, darkTheme: prismThemes.dracula},
  },
};

export default config;
```

## Phụ lục B. Mẫu `tsconfig.json` rút gọn

```json
{
  "extends": "@docusaurus/tsconfig",
  "compilerOptions": {
    "baseUrl": ".",
    "ignoreDeprecations": "6.0"
  },
  "include": ["src/", "docs/", "docusaurus.config.ts"]
}
```

## Phụ lục C. Checklist tái sử dụng cho agent mới

Khi một agent mới muốn tạo curriculum cho thư viện khác, copy skill này và thay thế các placeholder theo bảng ở đầu file. Không cần đọc lại toàn bộ; chỉ cần:

1. Điền bảng biến môi trường.
2. Chạy Pha A để clone `{TARGET_LIBRARY_NAME}`.
3. Chạy Pha B-I tuần tự.
4. Chạy Chương 5 để verify.
5. Commit + push lên remote.

Nếu gặp lỗi không có trong Chương 4, **bổ sung vào Chương 4** trước khi commit skill lên version mới, để agent sau không mất thời gian debug lại.

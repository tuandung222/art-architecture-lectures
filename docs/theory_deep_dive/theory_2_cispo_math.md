---
sidebar_position: 3
sidebar_label: "Theory 2: CISPO Math"
---

# Theory 2: CISPO Math: Clipped IS-weight Policy Optimization

CISPO là biến thể policy gradient đề xuất trong Mixture-of-Experts paper (sau đó được dùng trong nhiều hệ thống RL agentic production). Bài này phân tích tại sao CISPO ổn định hơn PPO cho agentic RL, và ánh xạ sang code ART.

---

## 1. PPO: công thức chuẩn

PPO (Schulman 2017):

\[
L^{PPO}(\theta) = -\mathbb{E}_{(s,a)\sim\pi_{\text{old}}} \left[ \min\left( \rho_\theta A,\; \text{clip}(\rho_\theta, 1-\varepsilon, 1+\varepsilon) A \right) \right]
\]

với \(\rho_\theta = \pi_\theta(a|s) / \pi_{\text{old}}(a|s)\).

Hai term `min` và `clip` tạo ra **3 vùng**:

1. **Nội clip range** (\(1-\varepsilon \le \rho \le 1+\varepsilon\)): cả hai term bằng \(\rho A\). Gradient chảy qua \(\rho\) và qua `log π` (qua \(\rho\) dưới dạng \(\partial \rho / \partial \theta\)).
2. **Ngoài clip + A > 0**: term `min` là `clip(...) A`, term kia là \(\rho A\). Min chọn `clip(...) A` (vì \(\rho A > \text{clip} A\)). Gradient chỉ chảy qua `clip` (hằng số), không qua \(\rho\).
3. **Ngoài clip + A < 0**: tương tự, chọn `clip(...) A` (vì `clip A > \rho A`, dương hơn). Gradient qua `clip`.

Hệ quả: PPO **cho phép gradient chảy qua \(\rho\)** khi ratio nằm trong dải clip. Nghĩa là nếu policy "tự tin" với action (ratio lớn), gradient được scale up.

---

## 2. Vấn đề của PPO với agentic RL

Hai vấn đề đặc biệt nghiêm trọng trong agentic RL:

### 2.1. Token-level ratio variance cao

Với rollout dài (1000+ token), \(\rho_\theta(a_t|s_t)\) dao động mạnh giữa các token. Một số token có ratio = 0.001 (model rất không tự tin so với old), một số = 100.0. PPO clip \(\rho\) ở dải [0.8, 1.2] (với \(\varepsilon = 0.2\)) sẽ:

* Cho ratio 0.001 -> 0.8 (scale up 800x).
* Cho ratio 100.0 -> 1.2 (scale down 80x).

Nhưng **gradient** vẫn chảy qua \(\rho\) trong dải clip. Nghĩa là: token có ratio = 1.05 (gần 1) có gradient tự nhiên; token có ratio = 0.85 (sau clip = 0.85) cũng có gradient tự nhiên; nhưng token có ratio = 0.5 (sau clip = 0.8) bị **scale up** đột ngột.

### 2.2. Importance-sampling hiệu quả kém

Trong PPO, `min` operator tạo ra một "trust region" ngầm. Khi policy lệch xa khỏi \(\pi_{\text{old}}\), một số token bị clip và contribution giảm. Nhưng effect không đối xứng giữa **advantage dương** và **advantage âm**:

* A > 0, ratio > 1+ε: clip làm giảm update (an toàn).
* A > 0, ratio < 1-ε: clip làm tăng update ngược chiều có lợi (vẫn OK).
* A < 0, ratio > 1+ε: clip "khoá" policy không cho rời xa hơn (an toàn).
* A < 0, ratio < 1-ε: clip có thể vô hiệu hóa update (mất tín hiệu).

Trong agentic RL với rollout dài, tỉ lệ token có A < 0 (rollout "xấu") thường lớn -> PPO mất nhiều gradient.

---

## 3. CISPO: công thức

CISPO (đề xuất trong context MoE training, OpenAI 2024):

\[
L^{CISPO}(\theta) = -\mathbb{E}_{(s,a)\sim\pi_{\text{old}}} \left[ \text{clip}(\rho_\theta, 1-\varepsilon, 1+\varepsilon) \cdot A \cdot \log \pi_\theta(a|s) \right]
\]

Hai khác biệt so với PPO:

1. **Clip \(\rho\) rồi detach**: ratio được clip và `detach()`, chỉ đóng vai trò **weight** không có gradient.
2. **Log pi thoáng**: gradient chỉ chảy qua \(\log \pi_\theta\), không qua \(\rho\).

Trong code (`src/art/loss.py`):

```python
if ppo:
    policy_loss = -torch.min(
        prob_ratio * advantages,
        torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
    )
else:
    # Modified REINFORCE or Clipped IS-weight Policy Optimization (CISPO)
    policy_loss = -(
        torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
        * advantages
        * new_logprobs
    )
```

---

## 4. Phân tích gradient

### 4.1. Gradient của CISPO

\[
\frac{\partial L}{\partial \theta} = -\mathbb{E}\left[ \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon) \cdot A \cdot \frac{\partial \log \pi_\theta}{\partial \theta} \right]
\]

Để ý: `clip(\rho)` là hằng số w.r.t. \(\theta\) (vì `detach()`). Gradient chỉ chảy qua `log π_θ`. Đây là REINFORCE với weight = clipped importance ratio.

### 4.2. So sánh với PPO

PPO gradient trong dải clip (vùng 1 ở trên):

\[
\frac{\partial L^{PPO}}{\partial \theta} = -\mathbb{E}\left[ A \cdot \rho \cdot \frac{\partial \log \pi_\theta}{\partial \theta} + A \cdot \frac{\partial \rho}{\partial \theta} \right]
\]

Term thứ hai (\(A \cdot \partial \rho / \partial \theta\)) tạo ra **gradient amplification** không mong muốn. Nếu policy đang lệch mạnh khỏi \(\pi_{\text{old}}\) mà vẫn trong dải clip (hiếm nhưng có thể xảy ra ở training đầu), gradient bị scale up.

CISPO **detach \(\rho\)** -> term thứ hai bằng 0. Gradient chỉ phụ thuộc vào `log π_θ` và advantage. An toàn hơn.

### 4.3. Hệ quả cho importance sampling

Mục đích của IS weight là **điều chỉnh phân phối sampling** từ \(\pi_{\text{old}}\) sang \(\pi_\theta\). Nếu \(\pi_\theta\) lệch, IS weight lớn để "phạt" sample đó. Nhưng trong PPO, IS weight cũng **ảnh hưởng gradient** -> mâu thuẫn.

CISPO tách bạch:

* IS weight điều chỉnh **magnitude** của update.
* Log pi điều khiển **hướng** của update.

Đây là lý do CISPO ổn định hơn khi rollout dài (variance IS weight cao).

---

## 5. Tại sao clip \(\rho\) quan trọng

Nếu không clip, IS weight có thể bùng nổ:

* \(\pi_{\text{old}} = 0.001\), \(\pi_\theta = 0.5\) -> \(\rho = 500\).
* Advantage A = 1.
* Update magnitude: \(500 \times 1 \times \log \pi_\theta\) -> loss có thể overflow.

CISPO clip \(\rho\) ở `1 + ε_H` (mặc định 4.0). Với clip 4.0, update magnitude bị giới hạn, gradient không overflow.

Nếu muốn clip chặt hơn (bảo thủ), dùng `epsilon_high=2.0`. Nếu muốn clip lỏng (cho phép update lớn), dùng `epsilon_high=10.0`.

```python
epsilon = experimental_config.get("epsilon", epsilon_default)         # 1.0
epsilon_high = experimental_config.get("epsilon_high", epsilon_high_default)  # 4.0
```

---

## 6. Token-level masking: `assistant_mask` và `weights`

CISPO trong `loss.py` cuối cùng:

```python
policy_loss = policy_loss * weights * assistant_mask
denominator = aligned_inputs.denominator(assistant_mask, reduction)
reduced_policy_loss = policy_loss.sum() / denominator
```

Hai mask quan trọng:

* `assistant_mask`: 1 cho token do assistant sinh, 0 cho token user/system/tool. Chỉ train trên token của model.
* `weights`: trọng số tùy ý (thường = 1). Có thể dùng để:
  * **Up-weight rollout quan trọng** (ví dụ rollout có advantage lớn).
  * **Down-weight rollout outlier** (ví dụ rollout dài bất thường).
  * **Multi-task**: weight = priority của task.

`denominator` chia theo tổng mask (`mean`) hoặc = 1 (`sum`). Mặc định `mean` để loss scale-invariant với batch size.

---

## 7. Ảnh hưởng của ε và ε_H

| ε | ε_H | Hành vi |
| --- | --- | --- |
| 0.2 | 0.2 | PPO-style: clip chặt, conservative |
| 1.0 | 4.0 | CISPO mặc định ART: cho phép IS weight lớn, gradient thoáng |
| 0.5 | 1.5 | Trung gian: an toàn hơn mặc định, nhanh hơn PPO |
| 0.0 | 0.0 | REINFORCE thuần (không clip), không ổn định |
| 5.0 | 20.0 | Gần như REINFORCE, rất nhanh nhưng dễ diverge |

Mẹo tuning:

* Bắt đầu với mặc định (1.0, 4.0).
* Nếu `probs_corr` (xem Bài 5 main) giảm nhanh (< 0.8 trong vài step): giảm `lr` hoặc tăng `epsilon` (chặt hơn ở dưới).
* Nếu `probs_corr` cao (> 0.99) và reward tăng chậm: giảm `epsilon` (cho phép update mạnh hơn khi ratio = 1).

---

## 8. PPO mode trong ART: khi nào cần

ART vẫn hỗ trợ `ppo=True` cho ai muốn dùng value network custom:

```python
config = dev.TrainConfig(ppo=True, epsilon=0.2, epsilon_high=0.2)
```

Khi nào cần PPO thay CISPO:

* Có value network riêng (tự train hoặc từ paper khác).
* Cần so sánh với baseline paper dùng PPO.
* Domain cần conservative update (ví dụ safety-critical).

Trong hầu hết case agentic, **CISPO tốt hơn** vì không cần value network và ổn định hơn.

---

## 9. Hệ quả với memory và throughput

Vì CISPO không cần value network:

* Memory: giảm ~50% (chỉ giữ policy, không giữ value).
* Compute: giảm ~30% (một forward pass thay vì hai).
* Wall-clock: training nhanh hơn 1.3-1.5x.

Đây là lý do ART chọn CISPO mặc định: tối ưu cho iteration speed trong research.

---

## 10. Minh họa: cùng rollout, CISPO vs PPO

Rollout dài 100 token, advantage = [từng token] với A cao ở giữa (token quan trọng), A thấp ở đầu/cuối.

Token 50 (giữa rollout): \(\pi_{\text{old}} = 0.05\), \(\pi_\theta = 0.20\) -> \(\rho = 4.0\). A = 0.5.

* **PPO** (ε=0.2): clip \(\rho\) về 1.2. Update magnitude: \(1.2 \times 0.5 = 0.6\). Nhưng gradient còn có term từ \(\partial \rho / \partial \theta\) -> thực tế ~ 0.7-0.8.
* **CISPO** (ε=1.0, ε_H=4.0): clip \(\rho\) về 4.0 (giữ nguyên). Update magnitude: \(4.0 \times 0.5 = 2.0\). Gradient chỉ qua `log π_θ`.

CISPO cho update mạnh hơn 3-4x cho token quan trọng. Khi rollout dài, đây là điểm khác biệt lớn.

---

## 11. Kết nối với bài tiếp theo

Trong bài này ta đã thấy \(\rho = \pi_\theta / \pi_{\text{old}}\) ở mức **token**. Bài tiếp theo ([Theory 3: IS Levels](theory_3_is_levels)) sẽ phân tích 3 cách aggregate ratio khác (sequence, average, geometric) và bias-variance trade-off.

---

## 12. Tóm tắt

| Khía cạnh | PPO | CISPO |
| --- | --- | --- |
| Clip | Trên `min` của 2 term | Trên \(\rho\) (detached) |
| Gradient qua \(\rho\) | Có (khi trong dải clip) | Không (detach) |
| Value network | Cần | Không |
| Memory | Cao | Thấp |
| ε mặc định | 0.2 | 1.0 (dải dưới) |
| ε_H mặc định | 0.2 (cùng) | 4.0 (dải trên) |
| Phù hợp | Dense reward, on-policy chặt | Agentic RL, multi-turn |

ART mặc định dùng CISPO (`ppo=False`). Bạn có thể đổi sang PPO bằng `experimental_config["ppo"] = True`.

---

Tiếp theo: [Theory 3: IS Levels](theory_3_is_levels).

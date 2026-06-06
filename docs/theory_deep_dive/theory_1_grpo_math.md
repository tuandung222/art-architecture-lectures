---
sidebar_position: 2
sidebar_label: "Theory 1: GRPO Math"
---

# Theory 1: GRPO Math: Group-Relative Advantage

Bài này dành cho bạn muốn hiểu sâu công thức GRPO: tại sao dùng group statistics thay Critic, khi nào GRPO ước lượng advantage tốt/xấu, và ánh xạ sang code ART.

---

## 1. Bài toán: ước lượng advantage không có Critic

Trong PPO chuẩn, advantage được tính bằng **Generalized Advantage Estimation (GAE)**:

$$
\hat\{A\}_t = \sum_\{l=0\}^\{\infty\} (\gamma \lambda)^l \delta_\{t+l\}, \quad \delta_t = r_t + \gamma V(s_\{t+1\}) - V(s_t)
$$

Yêu cầu: **value network** $V_\phi(s)$ phải được train song song với policy. Value network:

* Cùng kích thước với policy (ví dụ 7B).
* Phải hội tụ để ước lượng đúng advantage.
* Tăng gấp đôi memory footprint.

GRPO (DeepSeekMath, 2024) đề xuất: bỏ value network, dùng **group statistics** làm baseline.

---

## 2. Công thức GRPO

Với mỗi prompt $p$, rollout $K$ lần. Ký hiệu rewards $r_1, r_2, \dots, r_K$.

$$
A_i = \frac\{r_i - \mu_G\}\{\sigma_G + \epsilon\}
$$

trong đó:

* $\mu_G = \frac\{1\}\{K\} \sum_\{j=1\}^K r_j$.
* $\sigma_G = \sqrt\{\frac\{1\}\{K\} \sum_\{j=1\}^K (r_j - \mu_G)^2\}$.
* $\epsilon$: hằng số tránh chia 0 (ART dùng `1e-8` qua `group_aggregate`).

Hệ quả quan trọng:

* **Zero-mean**: $\sum_i A_i = 0$. Mỗi group có "ngân sách gradient" bằng 0.
* **Constant shift invariance**: nếu ta cộng $c$ vào mọi $r_i$, $\mu_G$ tăng $c$, $A_i$ không đổi.
* **Scale shift**: nếu ta nhân mọi $r_i$ với $h>0$, $A_i$ nhân $h$ (vì $\sigma_G$ nhân $h$, tỉ số không đổi).

Điều thứ hai (constant shift invariance) là chìa khóa cho RULER: judge LLM có thể "lệch" hằng số, GRPO vẫn cho cùng gradient.

---

## 3. Schulman unbiased estimator

Để GRPO hoạt động trong regime importance-sampling (rollout từ policy cũ, train policy mới), ta cần advantage estimator **unbiased** cho expected return. Schulman (2020, ICML) chứng minh rằng:

$$
\mathbb\{E\}_\{\tau \sim \pi_\{\text\{old\}\}\} \left[ \frac\{\pi_\theta(a|s)\}\{\pi_\{\text\{old\}\}(a|s)\} A^\{\pi_\{\text\{old\}\}\}(s,a) \right] = J(\pi_\theta) - J(\pi_\{\text\{old\}\})
$$

Trong đó $A^\{\pi_\{\text\{old\}\}\}$ là advantage của policy cũ. Nếu GRPO advantage $A_i$ là **unbiased estimator** của $A^\{\pi_\{\text\{old\}\}\}$, thì toàn bộ policy gradient vẫn đúng.

Khi nào $A_i$ là unbiased?

* Nếu rollout phân phối đúng theo $\pi_\{\text\{old\}\}$ và reward function không phụ thuộc vào $\theta$.
* Với cỡ mẫu $K$ đủ lớn, $\mu_G, \sigma_G$ hội tụ về expectation.

Khi nào $A_i$ **lệch**:

* Khi rollout từ policy rất khác $\pi_\{\text\{old\}\}$ (multi-step trên cùng batch).
* Khi reward có noise không zero-mean.

Trong ART, multi-step được giải quyết bằng **TIS** (Truncated Importance Sampling, xem Bài 4 về KL).

---

## 4. Variance và bias-variance trade-off

Vì GRPO dùng empirical mean/std thay vì true expectation, advantage có **variance**:

$$
\text\{Var\}(A_i) = \frac\{1\}\{K\} \text\{Var\}_\{\text\{rollout\}\}\left( \frac\{r_i - \mu_G\}\{\sigma_G\} \right) \approx \frac\{c\}\{K\}
$$

cho một số hằng số $c$ phụ thuộc vào reward distribution.

**Trade-off**:

* $K$ lớn -> variance nhỏ, nhưng tốn GPU time cho rollout.
* $K$ nhỏ -> variance lớn, training có thể dao động.

ART cho phép chọn $K$ qua `group_rollout_fn` (trong `pipeline_trainer`):

```python
def make_group_rollout_fn(single_rollout_fn, n=4):
    async def group_rollout(model, scenario, config):
        results = await asyncio.gather(
            *[single_rollout_fn(model, scenario, config) for _ in range(n)],
            return_exceptions=True,
        )
        return TrajectoryGroup(results)
    return group_rollout
```

Trong thực tế, $K = 4$ (Qwen 0.5B) đến $K = 18$ (Qwen 2.5 3B 2048) là phổ biến.

---

## 5. Code ART: `_backend_training.compute_group_advantages`

Trong `src/art/_backend_training.py`:

```python
def build_rl_train_configs(
    trajectory_groups: list[TrajectoryGroup],
    ...,
) -> ...:
    ...
```

Hàm này build danh sách các training example với advantage đã chuẩn hóa. Mỗi backend (LocalBackend, ServerlessBackend, TinkerBackend) implement riêng cách tính advantage. Pattern chung:

1. Với mỗi `TrajectoryGroup`, tính `rewards = [t.reward for t in group.trajectories]`.
2. Tính `mean = sum(rewards) / len(rewards)`.
3. Tính `var = sum((r - mean) ** 2 for r in rewards) / len(rewards)`.
4. `std = sqrt(var + eps)`.
5. `advantages = [(r - mean) / std for r in rewards]`.

Trong `LocalBackend._prepare_backend_for_training` (xem source), advantage được đưa vào `train_inputs` cho Unsloth.

---

## 6. Khi nào GRPO tốt hơn PPO

| Tiêu chí | GRPO | PPO |
| --- | --- | --- |
| Yêu cầu value network | Không | Có |
| Memory | Thấp hơn ~50% | Cao |
| Hyperparameter | Ít (chỉ K) | Nhiều (γ, λ, GAE params) |
| On-policy | Cứng (cần rollout mới mỗi step) | Linh hoạt (replay buffer) |
| Khi reward thưa thớt | Kém (cả group có thể reward = 0) | Tốt hơn (value network fit) |
| Khi reward dense | Tốt | Tốt |
| Multi-turn agent | Rất phù hợp | Cần GAE qua nhiều turn (phức tạp) |

Vì agentic RL thường có reward **thưa** (chỉ ở cuối trajectory) và **episode-level** (không per-step), GRPO + judge (RULER) là lựa chọn tự nhiên.

---

## 7. GRPO với reward thưa: hệ quả

Giả sử group có K=4 rollout, 3 rollout có reward 0 (fail), 1 rollout có reward 1 (success).

* $\mu_G = 0.25$
* $\sigma_G = 0.43$
* Advantage: success = (1 - 0.25) / 0.43 ≈ 1.74, fail = -0.58

Gradient chỉ có 1 hướng rõ ràng (token của rollout thành công). Nếu K=4 quá nhỏ, variance lớn. Workarounds:

* **Tăng K** đến 8-16: variance giảm $\sqrt\{2\}$-$2$.
* **Filter rollout fail**: bỏ rollout reward=0 trước khi tính advantage. Cẩn thận: nếu mọi rollout fail, group rỗng.
* **RULER judge**: relative scoring thay vì binary; rollout fail có reward 0.1-0.3 thay vì 0, rollout trung bình 0.5, rollout tốt 0.8. Advantage vẫn có ý nghĩa.

---

## 8. Minh họa bằng số

Group 4 rollout, rewards [0, 0, 1, 1]:

* $\mu_G = 0.5$, $\sigma_G = 0.5$.
* Advantage: 2 rollout đầu = -1.0, 2 rollout sau = +1.0.

Cùng rewards nhưng K=8 (4 rollout thêm với rewards [0, 1, 0, 1]):

* $\mu_G = 0.5$, $\sigma_G = 0.5$ (mean và std không đổi, do symmetric).
* Advantage giống hệt.

Nếu rollout thêm có rewards [0, 0, 0, 0]:

* $\mu_G = 0.25$, $\sigma_G = 0.43$.
* Advantage của rollout đầu: $A_0 = (0 - 0.25) / 0.43 = -0.58$. Nhỏ hơn magnitude 1.0 trước đó.

Hệ quả: **cùng reward, K khác nhau -> advantage khác nhau**. Đây là lý do K phải cố định qua các step (hoặc normalize lại).

---

## 9. GRPO với judge LLM nhiễu

Giả sử RULER judge thêm noise $\eta \sim \mathcal\{N\}(0, \sigma_j^2)$ vào mỗi score. Reward quan sát: $\hat\{r\}_i = r_i + \eta_i$.

* $\hat\{\mu\}_G = \mu_G + \bar\{\eta\}$ (mean noise).
* $\hat\{\sigma\}_G^2 = \sigma_G^2 + \sigma_j^2 + 2 \text\{Cov\}(r, \eta) \approx \sigma_G^2 + \sigma_j^2$ (nếu r và $\eta$ độc lập).
* Advantage: $\hat\{A\}_i = \frac\{r_i + \eta_i - \hat\{\mu\}_G\}\{\hat\{\sigma\}_G\}$.

Hệ quả:

* **Constant shift invariance vẫn đúng** (mean noise bị cancel).
* **Noise inflate denominator** -> advantage magnitude giảm. ART tự điều chỉnh bằng learning rate.
* **Rank preserved**: nếu noise nhỏ, thứ tự rollout không đổi. GRPO vẫn đúng hướng gradient.

Đây là lý do RULER + GRPO robust: judge LLM nhiễu nhưng vẫn cho gradient đúng.

---

## 10. Bias của empirical std

Có một nuance: dùng empirical std (chia K) thay vì unbiased std (chia K-1) làm advantage estimator **biased nhẹ**. Khi K nhỏ, hệ số điều chỉnh:

$$
\text\{std\}_\{\text\{unbiased\}\} = \sqrt\{\frac\{1\}\{K-1\} \sum (r - \mu)^2\} = \text\{std\}_\{\text\{empirical\}\} \cdot \sqrt\{\frac\{K\}\{K-1\}\}
$$

ART dùng empirical std (chia K) cho đơn giản. Khi K lớn (>10), sai số < 5%. Khi K=4, sai số ~8%, có thể amplify advantage magnitude. Workaround: tăng K hoặc scale advantage lên $\sqrt\{K/(K-1)\}$.

---

## 11. Tổng kết

GRPO có những đặc tính:

* **Ưu**: đơn giản, không cần value network, robust với judge noise.
* **Nhược**: yêu cầu K rollout mỗi prompt, variance ~ 1/sqrt(K), nhạy với reward thưa.
* **Khi nào dùng**: agentic RL với reward ở cuối episode, multi-turn, có judge LLM.
* **Khi nào KHÔNG dùng**: dense per-step reward (dùng PPO với value network sẽ hiệu quả hơn), sample budget rất hạn chế (K dưới 4).

ART implement GRPO thông qua:

1. `TrajectoryGroup` (Bài 4 main): gom K rollout cùng prompt.
2. `compute_group_advantages` (trong mỗi backend): chuẩn hóa advantage.
3. `loss_fn` (Bài 5 main): nhận advantage đã chuẩn hóa, tính gradient.

---

## 12. Bài tập

1. **Implement Toy GRPO** (tham khảo [Bài 8 toy pipeline](../lesson_8_pipeline_trainer_toy)): viết hàm `compute_advantages(rewards)` chuẩn hóa zero-mean unit-std.
2. **Vẽ đồ thị advantage vs K**: cùng rewards, thay K=2, 4, 8, 16. Quan sát variance.
3. **Test với noise**: thêm noise Gaussian vào rewards, đo correlation giữa advantage "clean" và "noisy".

---

Tiếp theo: [Theory 2: CISPO Math](theory_2_cispo_math) - cách ART dùng CISPO thay PPO để giảm gradient nhiễu từ importance ratio.

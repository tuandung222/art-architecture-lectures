import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero custom-hero', styles.heroBanner)}>
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'inline-flex', padding: '4px 12px', background: 'rgba(139, 92, 246, 0.15)', color: '#a78bfa', borderRadius: '100px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(139, 92, 246, 0.25)', marginBottom: '1.5rem' }}>
          AGENT REINFORCEMENT TRAINER (ART) DEEP DIVE
        </div>
        <Heading as="h1" className="hero__title" style={{ fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(135deg, #fff 30%, #a78bfa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '1.5rem' }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{ maxWidth: '750px', margin: '0 auto 2.5rem auto', fontSize: '1.2rem', lineHeight: '1.6', opacity: 0.85 }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', boxShadow: '0 4px 14px rgba(139, 92, 246, 0.4)', transition: 'all 0.3s ease' }}
            to="/docs/roadmap">
            Bắt đầu học ngay 🚀
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', marginLeft: '1rem', border: '1px solid var(--card-border)' }}
            href="https://github.com/tuandung222/art-architecture-lectures">
            View on GitHub 🛠️
          </Link>
        </div>
      </div>
    </header>
  );
}

interface FeatureItem {
  title: string;
  badge: string;
  description: string;
}

const CorePillars: FeatureItem[] = [
  {
    title: 'Client/Server Split',
    badge: 'Programming Paradigm',
    description: 'Tách biệt hoàn toàn Client Python (TrainableModel, Trajectory) và Backend (LocalBackend với vLLM/Unsloth hoặc Serverless W&B), cho phép tái sử dụng rollout trong mọi ứng dụng async.',
  },
  {
    title: 'GRPO & CISPO Loss',
    badge: 'Training Algorithm',
    description: 'Cơ chế tính Advantage nhóm tương đối (group-relative) kết hợp Clipped IS-weight Policy Optimization, với 4 cấp importance sampling (token, sequence, average, geometric_average) và KL penalty có cấu hình.',
  },
  {
    title: 'vLLM Runtime + NCCL Transfer',
    badge: 'Infrastructure',
    description: 'Tích hợp ManagedVllmRuntime thông qua manifest SHA256, cơ chế NCCL weight broadcast và packed tensor giúp đồng bộ LoRA giữa trainer và inference engine cực nhanh, hỗ trợ quy mô mô hình lên tới hàng trăm tỷ tham số qua Megatron context parallel.',
  },
];

interface LectureItem {
  number: string;
  title: string;
  desc: string;
  path: string;
  category: 'Background' | 'Core Theory' | 'Deep Dive Code' | 'Optimization' | 'Practice';
}

const Lectures: LectureItem[] = [
  {
    number: 'Bài 0',
    title: 'Nền tảng GRPO & Agentic RL',
    desc: 'Tổng quan về Agent Reinforcement Trainer: lý do ra đời, kiến trúc tổng thể Client/Server, vòng lặp huấn luyện inference-then-train cốt lõi, các khái niệm nền tảng (Trajectory, RULER, Multi-turn).',
    path: '/docs/lesson_0_agent_rl_fundamentals',
    category: 'Background'
  },
  {
    number: 'Bài 1',
    title: 'Thách thức Kỹ thuật trong Agentic RL',
    desc: 'Tại sao các framework SFT/RL truyền thống không đáp ứng multi-turn agent? Chi phí bộ nhớ khi tải 3-4 mô hình, độ trễ mạng giữa inference và training, xử lý tool calls và stateful execution.',
    path: '/docs/lesson_1_agentic_rl_challenges',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2',
    title: 'Kiến trúc Client/Server & OpenAI-compatible API',
    desc: 'Phân tích TrainableModel, AsyncOpenAI proxy, Backend Protocol. Cách ART đạt tính tương thích OpenAI thông qua HTTPX patching và auto_trajectory context.',
    path: '/docs/lesson_2_client_server_architecture',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3',
    title: 'Hệ thống Backend: Local, Serverless, Tinker, Megatron',
    desc: 'So sánh và phân tích LocalBackend (Unsloth/vLLM), ServerlessBackend (W&B), TinkerBackend, TinkerNativeBackend và Megatron training. Khi nào dùng backend nào?',
    path: '/docs/lesson_3_backends_zoo',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4',
    title: 'Trajectory & TrajectoryGroup: Mô hình Dữ liệu Cốt lõi',
    desc: 'Khảo sát cấu trúc Trajectory (messages_and_choices, additional_histories, metrics, metadata), cách gather_trajectory_groups thu thập kết quả từ hàng nghìn rollout song song.',
    path: '/docs/lesson_4_trajectory_data_model',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 5',
    title: 'GRPO & CISPO Loss: Toán học và Hiện thực',
    desc: 'Đi sâu vào file loss.py: GRPO advantage, Clipped IS-weight Policy Optimization, 4 cấp importance sampling (token, sequence, average, geometric_average), KL penalty với estimator (new-ref).',
    path: '/docs/lesson_5_grpo_cispo_loss',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6',
    title: 'vLLM Runtime, NCCL Weight Transfer & LoRA Hot-Swap',
    desc: 'Cơ chế ManagedVllmRuntime, VllmRuntimeManifest, TrainerNcclCommunicator và packed broadcast giúp đồng bộ LoRA giữa trainer và inference engine với độ trễ cực thấp.',
    path: '/docs/lesson_6_vllm_runtime_and_weight_transfer',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 7',
    title: 'RULER: Relative Universal LLM-Elicited Rewards',
    desc: 'Phân tích cơ chế LLM-as-judge tương đối của ART, common prefix token savings, judge rubric mặc định, tích hợp với after_each callback trong gather_trajectory_groups.',
    path: '/docs/lesson_7_ruler_and_reward_design',
    category: 'Optimization'
  },
  {
    number: 'Bài 8',
    title: 'Thực hành: Tự xây dựng Toy ART Pipeline',
    desc: 'Tự tay tái hiện pipeline 3 giai đoạn (rollout -> train -> eval) bằng Python thuần với asyncio, mô phỏng TrainableModel rút gọn và chạy cập nhật CISPO từ con số 0.',
    path: '/docs/lesson_8_pipeline_trainer_toy',
    category: 'Practice'
  }
];

function CategoryBadge({ category }: { category: LectureItem['category'] }) {
  const colors: Record<LectureItem['category'], { bg: string, text: string }> = {
    'Background': { bg: 'rgba(59, 130, 246, 0.15)', text: '#60a5fa' },
    'Core Theory': { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399' },
    'Deep Dive Code': { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24' },
    'Optimization': { bg: 'rgba(236, 72, 153, 0.15)', text: '#f472b6' },
    'Practice': { bg: 'rgba(139, 92, 246, 0.15)', text: '#a78bfa' },
  };

  const color = colors[category];

  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: color.bg, color: color.text, alignSelf: 'flex-start' }}>
      {category}
    </span>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} | Deep Dive ART Agent Reinforcement Trainer`}
      description="Chuỗi bài giảng phân tích chi tiết kiến trúc, thuật toán GRPO/CISPO và mã nguồn của thư viện Agent Reinforcement Trainer (ART) dành cho AI Engineers & RL Researchers.">
      <HomepageHeader />

      <main style={{ padding: '4rem 0', background: 'var(--ifm-background-color)' }}>
        {/* Core Pillars Section */}
        <section className="container" style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              🚀 Ba Trụ Cột Kiến Trúc Của ART
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Những đột phá thiết kế giúp ART trở thành framework RL gọn nhẹ nhất cho multi-step LLM agents
            </p>
          </div>

          <div className="row">
            {CorePillars.map((item, idx) => (
              <div key={idx} className="col col--4" style={{ marginBottom: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--ifm-color-primary)', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>
                    {item.badge}
                  </span>
                  <Heading as="h3" style={{ fontSize: '1.35rem', marginBottom: '1rem', fontWeight: 600 }}>
                    {item.title}
                  </Heading>
                  <p style={{ opacity: 0.8, fontSize: '0.95rem', lineHeight: '1.6', margin: 0 }}>
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Lectures Curriculum Section */}
        <section className="container">
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              📚 Giáo Trình Học Tập (Curriculum)
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Đi từ nền tảng lý thuyết GRPO, phân tích kiến trúc Client/Server, khảo sát mã nguồn loss và vLLM runtime, đến thực hành xây dựng pipeline hoàn chỉnh.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {Lectures.map((lecture, idx) => (
              <Link
                to={lecture.path}
                key={idx}
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <div className="glass-panel" style={{ padding: '1.5rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', cursor: 'pointer' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--ifm-color-primary)' }}>
                        {lecture.number}
                      </span>
                      <CategoryBadge category={lecture.category} />
                    </div>
                    <Heading as="h3" style={{ fontSize: '1.15rem', marginBottom: '0.75rem', fontWeight: 600, lineHeight: '1.4' }}>
                      {lecture.title}
                    </Heading>
                    <p style={{ opacity: 0.8, fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
                      {lecture.desc}
                    </p>
                  </div>
                  <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', fontSize: '0.85rem', fontWeight: 600, color: 'var(--ifm-color-primary-light)' }}>
                    Đọc bài học này <span style={{ marginLeft: '4px', transition: 'transform 0.2s ease' }} className="arrow-icon">→</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Practice Engine Banner */}
        <section className="container" style={{ marginTop: '6rem' }}>
          <div className="glass-panel" style={{ padding: '3rem', background: 'radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.12) 0%, transparent 60%), var(--card-bg)', textAlign: 'center', borderRadius: '16px' }}>
            <Heading as="h2" style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '1rem' }}>
              💻 Thực Hành: Tự Xây Dựng Toy ART Pipeline
            </Heading>
            <p style={{ maxWidth: '700px', margin: '0 auto 2rem auto', opacity: 0.8, lineHeight: '1.6' }}>
              Trong Bài 8, chúng ta sẽ tự tay triển khai bằng Python thuần một pipeline 3 giai đoạn (rollout -> train -> eval) của ART, mô phỏng TrainableModel rút gọn, gather_trajectory_groups và chạy cập nhật CISPO loss từ con số 0.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <Link
                className="button button--primary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600 }}
                to="/docs/lesson_8_pipeline_trainer_toy">
                Đến Bài Học Thực Hành
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}

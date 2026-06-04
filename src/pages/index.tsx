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
          DOCUMENTATION & PRACTICAL RLHF COURSE
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
            href="https://github.com/tuandung222/verl-architecture-lectures">
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
    title: 'HybridFlow Model',
    badge: 'Programming Paradigm',
    description: 'Tách biệt hoàn toàn luồng điều khiển (Control Flow - Single Process) và luồng tính toán (Computation Flow - Multi-process), tối ưu hóa khả năng mở rộng thuật toán và tái sử dụng mã nguồn.',
  },
  {
    title: '3D-HybridEngine',
    badge: 'Model Resharding',
    description: 'Cơ chế tự động chuyển đổi phân mảnh trọng số mô hình (sharding) cực nhanh giữa pha suy luận suy rộng (vLLM TP) và huấn luyện song song (FSDP/Megatron-LM), giảm thiểu hao phí VRAM.',
  },
  {
    title: 'Modular API Integrations',
    badge: 'Infrastructure',
    description: 'Tích hợp liền mạch với các hạ tầng phục vụ và huấn luyện LLM tiên tiến nhất như FSDP2, Megatron-LM, vLLM, và SGLang, hỗ trợ quy mô mô hình lên tới 671B tham số.',
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
    title: 'Nền tảng Alignment & Thuật toán PPO vs GRPO',
    desc: 'Hiểu bản chất của quá trình tinh chỉnh căn chỉnh (Alignment), chi tiết toán học và mô hình hóa thuật toán PPO và GRPO (Group Relative Policy Optimization).',
    path: '/docs/lesson_0_alignment_fundamentals',
    category: 'Background'
  },
  {
    number: 'Bài 1',
    title: 'Thách thức Kỹ thuật Hệ thống trong Distributed RLHF',
    desc: 'Tại sao các framework Deep Learning thông thường không đáp ứng tốt chu kỳ Rollout-Training phức tạp? Chi phí giao tiếp và nghẽn bộ nhớ khi tải 4 mô hình đồng thời.',
    path: '/docs/lesson_1_distributed_rl_challenges',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2',
    title: 'Mô hình Lập trình HybridFlow & Single-Controller',
    desc: 'Thiết kế Single Controller của verl dựa trên Ray. Khảo sát cách xây dựng WorkerGroup, ResourcePool và cơ chế đóng gói dữ liệu của DataProto.',
    path: '/docs/lesson_2_hybridflow_programming_model',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3',
    title: '3D-HybridEngine & Cơ chế Tự động Resharding Trọng số',
    desc: 'Phân tích cơ chế chuyển dịch sharding trọng số (NCCL-based) giữa FSDP/Megatron-LM và vLLM/SGLang. Tối ưu bộ nhớ đệm và giao tiếp.',
    path: '/docs/lesson_3_hybrid_engine_resharding',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4',
    title: 'Khảo sát Mã nguồn: Chu trình huấn luyện PPO & GRPO',
    desc: 'Đi sâu vào luồng thực thi trong main_ppo.py, RayPPOTrainer (fit) và core_algos.py. Theo sát đường truyền của dữ liệu từ khi sinh mẫu đến khi cập nhật mạng.',
    path: '/docs/lesson_4_source_code_walkthrough',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 5',
    title: 'Tích hợp & Tối ưu hóa Inference Engine (vLLM & SGLang)',
    desc: 'Cơ chế tích hợp vLLM/SGLang trong verl. Kỹ thuật Sequence Packing và Sequence Length Balancing nhằm phân phối tải tối ưu trên GPU.',
    path: '/docs/lesson_5_inference_engine_integration',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6',
    title: 'Quản lý Phần thưởng Khả thi & Tương tác Công cụ (Sandbox)',
    desc: 'Hiện thực Rule-based Reward cho các tác vụ lý luận (Math, Coding). Tích hợp môi trường Sandbox Fusion biệt lập và gọi công cụ đa lượt.',
    path: '/docs/lesson_6_verifiable_rewards_tool_use',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 7',
    title: 'Hướng dẫn Tối ưu Hiệu năng & Mở rộng Quy mô lớn',
    desc: 'Cấu hình tối ưu FSDP2 CPU Offloading, kỹ thuật LoRA RL tiết kiệm bộ nhớ, và cấu hình Expert Parallelism cho mô hình MoE siêu lớn (DeepSeek-671B).',
    path: '/docs/lesson_7_performance_tuning_scaling',
    category: 'Optimization'
  },
  {
    number: 'Bài 8',
    title: 'Thực hành: Tự xây dựng Toy RLHF Training Pipeline',
    desc: 'Tự tay lập trình một pipeline huấn luyện RLHF tối giản bằng Python: Mô phỏng Single-Controller, thiết kế DataProto rút gọn và chạy cập nhật GRPO/PPO.',
    path: '/docs/lesson_8_toy_rlhf_pipeline',
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
      title={`${siteConfig.title} | Deep Dive verl RLHF/RLAIF Architecture`}
      description="Chuỗi bài giảng phân tích chi tiết kiến trúc, thuật toán và mã nguồn của thư viện học tăng cường verl dành cho AI Alignment & Systems Engineers.">
      <HomepageHeader />
      
      <main style={{ padding: '4rem 0', background: 'var(--ifm-background-color)' }}>
        {/* Core Pillars Section */}
        <section className="container" style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              🚀 Ba Đột Phá Thiết Kế Của verl
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Những cột trụ kiến trúc giúp verl đạt hiệu năng vượt trội và tính linh hoạt tối đa khi huấn luyện RLHF
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
              Đi từ kiến thức nền tảng lý thuyết tinh chỉnh căn chỉnh, phân tích sâu các cơ chế hệ thống, cấu trúc mã nguồn đến thực hành viết pipeline.
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
              💻 Thực Hành: Tự Xây Dựng Toy RLHF Pipeline
            </Heading>
            <p style={{ maxWidth: '700px', margin: '0 auto 2rem auto', opacity: 0.8, lineHeight: '1.6' }}>
              Trong Bài 8, chúng ta sẽ tự tay triển khai bằng Python một chương trình mô phỏng kiến trúc điều khiển Single-Controller, cơ chế WorkerGroup chạy Ray giả lập, và luồng tính toán cập nhật GRPO/PPO từ con số 0.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <Link
                className="button button--primary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600 }}
                to="/docs/lesson_8_toy_rlhf_pipeline">
                Đến Bài Học Thực Hành
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}

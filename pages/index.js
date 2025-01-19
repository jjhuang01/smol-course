import Link from 'next/link'

export default function Home() {
  return (
    <div className="container">
      <main className="main">
        <h1 className="title">SMOL Course</h1>
        <p className="description">欢迎来到 SMOL Course 学习平台</p>
        <div className="grid">
          <Link href="/docs/学习指南" className="card">
            <div className="card-content">
              <h2>📚 学习指南 &rarr;</h2>
              <p>开始您的学习之旅</p>
            </div>
          </Link>
          <Link href="/docs/项目说明" className="card">
            <div className="card-content">
              <h2>📋 项目说明 &rarr;</h2>
              <p>了解项目详情</p>
            </div>
          </Link>
          <Link href="/docs/学习资料/AI关键词详解" className="card">
            <div className="card-content">
              <h2>📖 学习资料 &rarr;</h2>
              <p>浏览完整的学习资料</p>
            </div>
          </Link>
          <Link href="/docs/练习日志/解决方案/代码问题解决方案" className="card">
            <div className="card-content">
              <h2>📝 练习日志 &rarr;</h2>
              <p>查看问题解决方案</p>
            </div>
          </Link>
        </div>
      </main>

      <style jsx>{`
        .container {
          min-height: 100vh;
          padding: 0 0.5rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          background-color: var(--bg-color);
          color: var(--text-color);
        }

        .main {
          padding: 4rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          max-width: 1200px;
          width: 100%;
        }

        .title {
          margin: 0;
          line-height: 1.15;
          font-size: 4rem;
          text-align: center;
          background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          font-weight: bold;
          text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .description {
          margin: 2rem 0;
          line-height: 1.5;
          font-size: 1.5rem;
          text-align: center;
          color: var(--text-secondary);
        }

        .grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(300px, 1fr));
          gap: 2rem;
          max-width: 1000px;
          margin-top: 3rem;
          width: 100%;
          padding: 0 1rem;
        }

        .card {
          text-decoration: none;
          color: inherit;
        }

        .card-content {
          padding: 1.5rem;
          border-radius: 10px;
          background: var(--card-bg);
          border: 1px solid var(--border-color);
          transition: all 0.3s ease;
          height: 100%;
        }

        .card:hover .card-content {
          transform: translateY(-5px);
          box-shadow: 0 8px 30px rgba(0,0,0,0.12);
          border-color: var(--primary-color);
        }

        .card h2 {
          margin: 0 0 1rem 0;
          font-size: 1.5rem;
          color: var(--heading-color);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .card p {
          margin: 0;
          font-size: 1.1rem;
          line-height: 1.5;
          color: var(--text-secondary);
        }

        @media (max-width: 768px) {
          .title {
            font-size: 3rem;
          }

          .grid {
            grid-template-columns: 1fr;
            margin: 3rem 1rem;
          }

          .card {
            margin: 1rem 0;
          }
        }

        :root {
          --primary-color: #0070f3;
          --secondary-color: #00a6ed;
          --bg-color: #ffffff;
          --text-color: #1a202c;
          --text-secondary: #4a5568;
          --heading-color: #2d3748;
          --card-bg: #ffffff;
          --border-color: #e2e8f0;
        }

        :global(.dark) {
          --primary-color: #60a5fa;
          --secondary-color: #3b82f6;
          --bg-color: #1a202c;
          --text-color: #f7fafc;
          --text-secondary: #e2e8f0;
          --heading-color: #f7fafc;
          --card-bg: #2d3748;
          --border-color: #4a5568;
        }
      `}</style>
    </div>
  )
} 
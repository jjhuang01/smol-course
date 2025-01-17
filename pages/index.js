import Link from 'next/link'

export default function Home() {
  return (
    <div className="container">
      <main className="main">
        <h1 className="title">SMOL Course</h1>
        <p className="description">欢迎来到 SMOL Course 学习平台</p>
        
        <div className="grid">
          <Link href="/docs/学习指南" className="card">
            <h2>学习指南 &rarr;</h2>
            <p>开始您的学习之旅</p>
          </Link>

          <Link href="/docs/项目说明" className="card">
            <h2>项目说明 &rarr;</h2>
            <p>了解项目详情</p>
          </Link>

          <Link href="/docs/学习资料/AI关键词详解" className="card">
            <h2>学习资料 &rarr;</h2>
            <p>浏览完整的学习资料</p>
          </Link>

          <Link href="/docs/练习日志/解决方案/代码问题解决方案" className="card">
            <h2>练习日志 &rarr;</h2>
            <p>查看问题解决方案</p>
          </Link>
        </div>
      </main>

      <style jsx>{`
        .main {
          padding: 4rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }

        .title {
          margin: 0;
          line-height: 1.15;
          font-size: 4rem;
          text-align: center;
          background: linear-gradient(45deg, #0070f3, #00a6ed);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .description {
          margin: 2rem 0;
          line-height: 1.5;
          font-size: 1.5rem;
          text-align: center;
          color: #666;
        }

        .grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(300px, 1fr));
          gap: 2rem;
          max-width: 1000px;
          margin-top: 3rem;
        }

        .card {
          padding: 1.5rem;
          border-radius: 10px;
          background: #fff;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          transition: all 0.3s ease;
        }

        .card:hover {
          transform: translateY(-5px);
          box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .card h2 {
          margin: 0 0 1rem 0;
          font-size: 1.5rem;
          color: #0070f3;
        }

        .card p {
          margin: 0;
          font-size: 1.1rem;
          line-height: 1.5;
          color: #666;
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
      `}</style>
    </div>
  )
} 
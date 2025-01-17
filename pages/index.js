import Link from 'next/link'

export default function Home() {
  return (
    <div className="container">
      <main>
        <h1>SMOL Course</h1>
        <p>欢迎来到 SMOL Course 学习平台</p>
        
        <div className="grid">
          <Link href="/docs/学习指南">
            <div className="card">
              <h2>学习指南 &rarr;</h2>
              <p>开始您的学习之旅</p>
            </div>
          </Link>

          <Link href="/docs/项目说明">
            <div className="card">
              <h2>项目说明 &rarr;</h2>
              <p>了解项目详情</p>
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
        }

        main {
          padding: 5rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }

        .grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(200px, 1fr));
          gap: 1.5rem;
          max-width: 800px;
          margin-top: 3rem;
        }

        .card {
          padding: 1.5rem;
          border: 1px solid #eaeaea;
          border-radius: 10px;
          transition: color 0.15s ease, border-color 0.15s ease;
          cursor: pointer;
        }

        .card:hover {
          border-color: #0070f3;
        }
      `}</style>
    </div>
  )
} 
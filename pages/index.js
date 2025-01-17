import Link from 'next/link'

export default function Home() {
  return (
    <div style={{
      minHeight: '100vh',
      padding: '0 0.5rem',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      <main style={{
        padding: '5rem 0',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <h1>SMOL Course</h1>
        <p>欢迎来到 SMOL Course 学习平台</p>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(200px, 1fr))',
          gap: '1.5rem',
          maxWidth: '800px',
          marginTop: '3rem'
        }}>
          <Link href="/docs/学习指南" style={{ textDecoration: 'none', color: 'inherit' }}>
            <div style={{
              padding: '1.5rem',
              border: '1px solid #eaeaea',
              borderRadius: '10px',
              transition: 'border-color 0.15s ease',
              cursor: 'pointer'
            }}>
              <h2>学习指南 &rarr;</h2>
              <p>开始您的学习之旅</p>
            </div>
          </Link>

          <Link href="/docs/项目说明" style={{ textDecoration: 'none', color: 'inherit' }}>
            <div style={{
              padding: '1.5rem',
              border: '1px solid #eaeaea',
              borderRadius: '10px',
              transition: 'border-color 0.15s ease',
              cursor: 'pointer'
            }}>
              <h2>项目说明 &rarr;</h2>
              <p>了解项目详情</p>
            </div>
          </Link>
        </div>
      </main>
    </div>
  )
} 
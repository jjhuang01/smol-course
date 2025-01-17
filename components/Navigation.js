import Link from 'next/link'
import { useRouter } from 'next/router'

const menuItems = [
  {
    title: '学习指南',
    path: '/docs/学习指南',
  },
  {
    title: '项目说明',
    path: '/docs/项目说明',
  },
  {
    title: '学习资料',
    items: [
      { title: 'AI关键词详解', path: '/docs/学习资料/AI关键词详解' },
      { title: 'AI前沿技术详解', path: '/docs/学习资料/AI前沿技术详解' },
      { title: 'AI实战项目指南', path: '/docs/学习资料/AI实战项目指南' },
      { title: 'PyTorch深度学习基础', path: '/docs/学习资料/PyTorch深度学习基础' },
      { title: 'Python数据科学基础', path: '/docs/学习资料/Python数据科学基础' },
      { title: '统计学详解', path: '/docs/学习资料/统计学详解' },
      { title: '假设检验详解', path: '/docs/学习资料/假设检验详解' },
    ]
  },
  {
    title: '练习日志',
    items: [
      { title: '代码问题解决方案', path: '/docs/练习日志/解决方案/代码问题解决方案' },
      { title: '模型问题解决方案', path: '/docs/练习日志/解决方案/模型问题解决方案' },
      { title: '环境问题解决方案', path: '/docs/练习日志/解决方案/环境问题解决方案' },
    ]
  }
]

export default function Navigation() {
  const router = useRouter()
  
  return (
    <nav className="navigation">
      <div className="nav-content">
        <Link href="/" className="logo">
          SMOL Course
        </Link>
        
        <div className="menu">
          {menuItems.map((item) => (
            <div key={item.title} className="menu-item">
              <span className="menu-title">{item.title}</span>
              {item.items ? (
                <div className="dropdown">
                  {item.items.map((subItem) => (
                    <Link
                      key={subItem.path}
                      href={subItem.path}
                      className={`dropdown-item ${router.asPath === subItem.path ? 'active' : ''}`}
                    >
                      {subItem.title}
                    </Link>
                  ))}
                </div>
              ) : (
                <Link
                  href={item.path}
                  className={router.asPath === item.path ? 'active' : ''}
                >
                  {item.title}
                </Link>
              )}
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .navigation {
          background: #1a1a1a;
          color: #fff;
          padding: 1rem 0;
          position: sticky;
          top: 0;
          z-index: 100;
        }

        .nav-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 1rem;
          display: flex;
          align-items: center;
        }

        .logo {
          font-size: 1.5rem;
          font-weight: bold;
          color: #fff;
          text-decoration: none;
          margin-right: 2rem;
        }

        .menu {
          display: flex;
          gap: 1.5rem;
        }

        .menu-item {
          position: relative;
          padding: 0.5rem 0;
        }

        .menu-title {
          cursor: pointer;
          color: #fff;
        }

        .dropdown {
          display: none;
          position: absolute;
          top: 100%;
          left: 0;
          background: #2a2a2a;
          border-radius: 4px;
          padding: 0.5rem 0;
          min-width: 200px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        .menu-item:hover .dropdown {
          display: block;
        }

        .dropdown-item {
          display: block;
          padding: 0.5rem 1rem;
          color: #fff;
          text-decoration: none;
          transition: background-color 0.2s;
        }

        .dropdown-item:hover {
          background: #3a3a3a;
        }

        .active {
          color: #0070f3;
        }

        @media (max-width: 768px) {
          .nav-content {
            flex-direction: column;
            align-items: flex-start;
          }

          .menu {
            flex-direction: column;
            width: 100%;
            margin-top: 1rem;
          }

          .dropdown {
            position: static;
            display: none;
            width: 100%;
          }

          .menu-item:hover .dropdown {
            display: block;
          }
        }
      `}</style>
    </nav>
  )
} 
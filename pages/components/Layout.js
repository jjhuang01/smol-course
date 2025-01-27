import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

export default function Layout({ children }) {
  const [searchQuery, setSearchQuery] = useState('');
  const router = useRouter();

  // 生成面包屑导航
  const generateBreadcrumbs = () => {
    const pathSegments = router.asPath.split('/').filter(segment => segment);
    let breadcrumbs = [];
    let path = '';

    pathSegments.forEach((segment, index) => {
      path += `/${segment}`;
      breadcrumbs.push({
        text: decodeURIComponent(segment),
        href: path,
        isLast: index === pathSegments.length - 1
      });
    });

    return breadcrumbs;
  };

  // 处理搜索
  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchQuery)}`);
    }
  };

  // 导航菜单新增条目
  const menuItems = [
    {
      path: '/docs/基础入门',
      name: '🥘 新手厨房',
      subItems: [
        { path: '/docs/学习资料/AI学习路径指南', name: '📖 入门食谱' },
        { path: '/docs/学习资料/生活化AI实验室', name: '🧪 实验厨房' },
        { path: '/docs/学习资料/AI关键词详解', name: '🔍 厨具图鉴' },
        { path: '/docs/学习资料/统计学入门：像讲故事一样学习', name: '📊 数学料理' }
      ]
    },
    {
      path: '/docs/进阶学习',
      name: '👨‍🍳 进阶课堂',
      subItems: [
        { path: '/docs/学习资料/机器学习算法详解', name: '🍳 烹饪技法' },
        { path: '/docs/学习资料/深度学习中的注意力机制详解', name: '🎯 专注力艺术' },
        { path: '/docs/学习资料/强化学习基础与实践', name: '🎮 厨艺游戏' },
        { path: '/docs/学习资料/模型调优实战手册', name: '🛠️ 调味秘籍' }
      ]
    },
    {
      path: '/docs/专家进阶',
      name: '🎓 大厨修炼',
      subItems: [
        { path: '/docs/学习资料/AI前沿技术详解', name: '🔮 未来美食' },
        { path: '/docs/学习资料/大模型微调技术详解', name: '🎛️ 配方改良' },
        { path: '/docs/学习资料/AI模型部署与工程实践', name: '🏪 开店指南' }
      ]
    },
    {
      path: '/docs/实践指南',
      name: '🏆 实战演练',
      subItems: [
        { path: '/docs/实践指南/项目实战案例', name: '📋 真实案例' },
        { path: '/docs/实践指南/常见问题解决', name: '🔧 疑难解答' },
        { path: '/docs/学习资料/AI错题本', name: '📝 错题笔记' }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link href="/" className="text-xl font-bold text-gray-800">
                  🍽️ AI美食学院
                </Link>
              </div>
              
              <div className="hidden md:ml-6 md:flex md:space-x-8">
                {menuItems.map((item, index) => (
                  <div key={index} className="relative group">
                    <button className="px-3 py-2 text-gray-700 hover:text-gray-900">
                      {item.name}
                    </button>
                    <div className="absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg hidden group-hover:block">
                      {item.subItems.map((subItem, subIndex) => (
                        <Link 
                          key={subIndex}
                          href={subItem.path} 
                          className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                        >
                          {subItem.name}
                        </Link>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 搜索框 */}
            <div className="hidden md:flex items-center">
              <form onSubmit={handleSearch} className="flex items-center">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="🔍 搜索食谱..."
                  className="px-4 py-2 border rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-500 text-white rounded-r-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  搜索
                </button>
              </form>
            </div>

            {/* 移动端菜单按钮 */}
            <div className="md:hidden flex items-center">
              <button className="mobile-menu-button p-2 rounded-md hover:bg-gray-100 focus:outline-none">
                <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* 面包屑导航 */}
      <div className="max-w-7xl mx-auto px-4 py-2">
        <div className="flex items-center space-x-2 text-sm text-gray-600">
          <Link href="/" className="hover:text-gray-900">
            首页
          </Link>
          {generateBreadcrumbs().map((crumb, index) => (
            <React.Fragment key={crumb.href}>
              <span>/</span>
              {crumb.isLast ? (
                <span className="text-gray-900">{crumb.text}</span>
              ) : (
                <Link href={crumb.href} className="hover:text-gray-900">
                  {crumb.text}
                </Link>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {children}
      </main>

      <footer className="bg-white shadow-inner mt-8">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500 text-sm">
            © 2024 AI美食学院 🍳 最后更新: 2024-03-21
          </p>
        </div>
      </footer>
    </div>
  );
} 
import { useState, useEffect, Fragment } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { motion } from 'framer-motion'
import { Disclosure, Dialog, Transition } from '@headlessui/react'
import { ChevronRightIcon, Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import Prism from 'prismjs'
import ThemeToggle from './ThemeToggle'
import Search from './Search'
import Pagination from './Pagination'
import ProgressBar from './ProgressBar'
import TableOfContents from './TableOfContents'
import CodeBlock from './CodeBlock'

const menuItems = [
  {
    title: '学习指南',
    path: '/docs/学习指南',
    icon: '📚',
    index: '1',
  },
  {
    title: '项目说明',
    path: '/docs/项目说明',
    icon: '📋',
    index: '2',
  },
  {
    title: '学习资料',
    icon: '📖',
    index: '3',
    items: [
      { title: 'AI关键词详解', path: '/docs/学习资料/AI关键词详解', index: '3.1' },
      { title: 'AI前沿技术详解', path: '/docs/学习资料/AI前沿技术详解', index: '3.2' },
      { title: 'AI实战项目指南', path: '/docs/学习资料/AI实战项目指南', index: '3.3' },
      { title: 'PyTorch深度学习基础', path: '/docs/学习资料/PyTorch深度学习基础', index: '3.4' },
      { title: 'Python数据科学基础', path: '/docs/学习资料/Python数据科学基础', index: '3.5' },
      { title: '统计学详解', path: '/docs/学习资料/统计学详解', index: '3.6' },
      { title: '假设检验详解', path: '/docs/学习资料/假设检验详解', index: '3.7' },
      { title: '统计学基础知识', path: '/docs/学习资料/统计学基础知识', index: '3.8' },
      { title: '机器学习算法详解', path: '/docs/学习资料/机器学习算法详解', index: '3.9' },
      { title: '深度学习架构详解', path: '/docs/学习资料/深度学习架构详解', index: '3.10' },
      { title: '统计学重点内容详解', path: '/docs/学习资料/统计学重点内容详解', index: '3.11' },
      { title: '人工智能核心公式详解', path: '/docs/学习资料/人工智能核心公式详解', index: '3.12' },
      { title: '模型评估与概率模型详解', path: '/docs/学习资料/模型评估与概率模型详解', index: '3.13' },
      { title: '统计学入门：像讲故事一样学习', path: '/docs/学习资料/统计学入门：像讲故事一样学习', index: '3.14' },
      { title: '数据集详解：像生活一样理解数据', path: '/docs/学习资料/数据集详解：像生活一样理解数据', index: '3.15' }
    ]
  },
  {
    title: '练习日志',
    icon: '📝',
    index: '4',
    items: [
      { title: '练习日志说明', path: '/docs/练习日志/README', index: '4.1' },
      { title: '代码问题解决方案', path: '/docs/练习日志/解决方案/代码问题解决方案', index: '4.2' },
      { title: '模型问题解决方案', path: '/docs/练习日志/解决方案/模型问题解决方案', index: '4.3' },
      { title: '环境问题解决方案', path: '/docs/练习日志/解决方案/环境问题解决方案', index: '4.4' }
    ]
  },
  {
    title: '系统文档',
    icon: '📘',
    index: '5',
    items: [
      { title: '目录索引', path: '/docs/目录索引', index: '5.1' },
      { title: '文档更新规则', path: '/docs/文档更新规则', index: '5.2' },
      { title: '系统架构设计', path: '/docs/系统架构设计', index: '5.3' },
      { title: '问题解决指南', path: '/docs/问题解决指南', index: '5.4' }
    ]
  }
]

// 侧边栏内容组件
function SidebarContent() {
  const router = useRouter()

  const renderMenuItem = (item, depth = 0) => {
    // 解码路径进行比较
    const currentPath = decodeURIComponent(router.asPath)
    const itemPath = item.path ? decodeURIComponent(item.path) : null
    const isActive = currentPath === itemPath
    const hasSubItems = item.items && item.items.length > 0
    
    // 检查子项是否被选中
    const hasActiveChild = hasSubItems && item.items.some(subItem => 
      decodeURIComponent(subItem.path) === currentPath
    )
    
    return (
      <div key={item.title} className={`pl-${depth * 4}`}>
        {item.path ? (
          <Link
            href={item.path}
            className={`flex items-center px-4 py-2 text-sm font-medium ${
              isActive
                ? 'text-primary-600 bg-primary-50 dark:bg-primary-900/10 font-bold'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700'
            }`}
          >
            {item.icon && <span className="mr-2">{item.icon}</span>}
            <span className="mr-2 text-gray-400">{item.index}</span>
            <span>{item.title}</span>
          </Link>
        ) : (
          <Disclosure defaultOpen={hasActiveChild}>
            {({ open }) => (
              <>
                <Disclosure.Button
                  className={`flex items-center w-full px-4 py-2 text-sm font-medium ${
                    hasActiveChild
                      ? 'text-primary-600 bg-primary-50 dark:bg-primary-900/10 font-bold'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700'
                  } cursor-pointer`}
                >
                  {item.icon && <span className="mr-2">{item.icon}</span>}
                  <span className="mr-2 text-gray-400">{item.index}</span>
                  <span>{item.title}</span>
                  <ChevronRightIcon
                    className={`ml-auto h-4 w-4 transform transition-transform duration-200 ${
                      open ? 'rotate-90' : ''
                    }`}
                  />
                </Disclosure.Button>
                <Transition
                  enter="transition duration-100 ease-out"
                  enterFrom="transform scale-95 opacity-0"
                  enterTo="transform scale-100 opacity-100"
                  leave="transition duration-75 ease-out"
                  leaveFrom="transform scale-100 opacity-100"
                  leaveTo="transform scale-95 opacity-0"
                >
                  <Disclosure.Panel className="pl-4">
                    {item.items?.map((subItem) => renderMenuItem(subItem, depth + 1))}
                  </Disclosure.Panel>
                </Transition>
              </>
            )}
          </Disclosure>
        )}
        {hasSubItems && item.path && (
          <div className="ml-4">
            {item.items.map((subItem) => renderMenuItem(subItem, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <nav className="mt-5 flex-1 px-2 space-y-1">
      {menuItems.map((item) => renderMenuItem(item))}
    </nav>
  )
}

export default function Layout({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const router = useRouter()

  useEffect(() => {
    if (typeof window !== 'undefined') {
      Prism.highlightAll()
    }
  }, [children])

  // 获取当前页面的前后页面
  const getCurrentPageInfo = () => {
    const flattenedItems = [];
    const flatten = (items) => {
      items.forEach(item => {
        if (item.path) {
          flattenedItems.push({
            title: item.title,
            path: item.path,
            index: item.index
          });
        }
        if (item.items) {
          flatten(item.items);
        }
      });
    };
    flatten(menuItems);

    // 使用版本号比较算法来排序
    flattenedItems.sort((a, b) => {
      const aIndexParts = a.index.split('.').map(Number);
      const bIndexParts = b.index.split('.').map(Number);
      
      // 比较每一级的索引
      for (let i = 0; i < Math.max(aIndexParts.length, bIndexParts.length); i++) {
        const aValue = aIndexParts[i] || 0;
        const bValue = bIndexParts[i] || 0;
        if (aValue !== bValue) {
          return aValue - bValue;
        }
      }
      return 0;
    });

    // 解码当前路径进行比较
    const currentPath = decodeURIComponent(router.asPath);
    
    // 如果是根路径，返回第一个页面作为下一页
    if (currentPath === '/') {
      return {
        prevPage: null,
        nextPage: flattenedItems[0]
      };
    }
    
    const currentIndex = flattenedItems.findIndex(item => decodeURIComponent(item.path) === currentPath);
    
    // 确保找到了当前页面
    if (currentIndex === -1) {
      console.warn('Current page not found in menu items:', currentPath);
      return { prevPage: null, nextPage: null };
    }

    return {
      prevPage: currentIndex > 0 ? flattenedItems[currentIndex - 1] : null,
      nextPage: currentIndex < flattenedItems.length - 1 ? flattenedItems[currentIndex + 1] : null
    };
  };

  const { prevPage, nextPage } = getCurrentPageInfo();

  return (
    <div className="h-screen flex overflow-hidden">
      <ProgressBar />
      
      {/* Sidebar for mobile */}
      <Transition.Root show={isSidebarOpen} as={Fragment}>
        <Dialog as="div" className="fixed inset-0 flex z-40" onClose={setIsSidebarOpen}>
          <Transition.Child
            as={Fragment}
            enter="transition-opacity ease-linear duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="transition-opacity ease-linear duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Dialog.Overlay className="fixed inset-0 bg-gray-600 bg-opacity-75" />
          </Transition.Child>
          <Transition.Child
            as={Fragment}
            enter="transition ease-in-out duration-300 transform"
            enterFrom="-translate-x-full"
            enterTo="translate-x-0"
            leave="transition ease-in-out duration-300 transform"
            leaveFrom="translate-x-0"
            leaveTo="-translate-x-full"
          >
            <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white dark:bg-gray-800">
              <div className="absolute top-0 right-0 -mr-12 pt-2">
                <button
                  type="button"
                  className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                  onClick={() => setIsSidebarOpen(false)}
                >
                  <span className="sr-only">关闭侧边栏</span>
                  <XMarkIcon className="h-6 w-6 text-white" aria-hidden="true" />
                </button>
              </div>
              <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
                <div className="flex-shrink-0 flex items-center px-4">
                  <Link href="/" className="text-xl font-bold text-primary-600">
                    SMOL Course
                  </Link>
                </div>
                <SidebarContent />
              </div>
            </div>
          </Transition.Child>
        </Dialog>
      </Transition.Root>

      {/* Static sidebar for desktop */}
      <div className="hidden lg:flex lg:flex-shrink-0">
        <div className="flex flex-col w-64">
          <div className="flex flex-col min-h-0 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
              <div className="flex items-center flex-shrink-0 px-4">
                <Link href="/" className="text-xl font-bold text-primary-600">
                  SMOL Course
                </Link>
              </div>
              <SidebarContent />
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col w-0 flex-1 overflow-hidden">
        <div className="relative z-10 flex-shrink-0 flex h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <button
            type="button"
            className="px-4 border-r border-gray-200 dark:border-gray-700 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 lg:hidden"
            onClick={() => setIsSidebarOpen(true)}
          >
            <span className="sr-only">打开侧边栏</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>

          {/* Header tools */}
          <div className="flex-1 px-4 flex justify-end">
            <div className="ml-4 flex items-center space-x-4">
              <Search menuItems={menuItems} />
              <ThemeToggle />
            </div>
          </div>
        </div>

        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <div className="py-6">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 xl:pr-72">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.3 }}
                className="prose dark:prose-dark max-w-none"
              >
                {children}
                <Pagination prevPage={prevPage} nextPage={nextPage} />
              </motion.div>
            </div>
          </div>
        </main>
      </div>
      <TableOfContents />
    </div>
  )
} 
import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { motion, AnimatePresence } from 'framer-motion'
import { Disclosure, Transition } from '@headlessui/react'
import { ChevronRightIcon, Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

const menuItems = [
  {
    title: '学习指南',
    path: '/docs/学习指南',
    icon: '📚',
  },
  {
    title: '项目说明',
    path: '/docs/项目说明',
    icon: '📋',
  },
  {
    title: '学习资料',
    icon: '📖',
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
    icon: '📝',
    items: [
      { title: '代码问题解决方案', path: '/docs/练习日志/解决方案/代码问题解决方案' },
      { title: '模型问题解决方案', path: '/docs/练习日志/解决方案/模型问题解决方案' },
      { title: '环境问题解决方案', path: '/docs/练习日志/解决方案/环境问题解决方案' },
    ]
  }
]

export default function Layout({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const router = useRouter()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar */}
      <div className="lg:hidden">
        <div className="fixed inset-0 flex z-40">
          <Transition
            show={isSidebarOpen}
            enter="transition-opacity ease-linear duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="transition-opacity ease-linear duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0">
              <div className="absolute inset-0 bg-gray-600 opacity-75"></div>
            </div>
          </Transition>

          <Transition
            show={isSidebarOpen}
            enter="transition ease-in-out duration-300 transform"
            enterFrom="-translate-x-full"
            enterTo="translate-x-0"
            leave="transition ease-in-out duration-300 transform"
            leaveFrom="translate-x-0"
            leaveTo="-translate-x-full"
          >
            <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white">
              <div className="absolute top-0 right-0 -mr-12 pt-2">
                <button
                  className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                  onClick={() => setIsSidebarOpen(false)}
                >
                  <span className="sr-only">关闭侧边栏</span>
                  <XMarkIcon className="h-6 w-6 text-white" aria-hidden="true" />
                </button>
              </div>
              <SidebarContent />
            </div>
          </Transition>
          <div className="flex-shrink-0 w-14"></div>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:flex-shrink-0">
        <div className="flex flex-col w-64">
          <div className="flex flex-col flex-grow border-r border-gray-200 pt-5 pb-4 bg-white overflow-y-auto">
            <SidebarContent />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-1">
        <div className="sticky top-0 z-10 flex-shrink-0 flex h-16 bg-white shadow">
          <button
            className="px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 lg:hidden"
            onClick={() => setIsSidebarOpen(true)}
          >
            <span className="sr-only">打开侧边栏</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>
          <div className="flex-1 px-4 flex justify-between">
            <div className="flex-1 flex">
              <Link href="/" className="flex items-center">
                <span className="text-2xl font-bold text-primary-600">SMOL Course</span>
              </Link>
            </div>
          </div>
        </div>

        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            className="py-6"
          >
            <div className="max-w-4xl mx-auto px-4 sm:px-6 md:px-8">
              {children}
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  )
}

function SidebarContent() {
  const router = useRouter()

  return (
    <nav className="mt-5 flex-1 flex flex-col divide-y divide-gray-200 overflow-y-auto" aria-label="Sidebar">
      <div className="px-2 space-y-1">
        {menuItems.map((item) => (
          <div key={item.title}>
            {item.items ? (
              <Disclosure defaultOpen={item.items.some(subItem => router.asPath === subItem.path)}>
                {({ open }) => (
                  <>
                    <Disclosure.Button className="w-full flex items-center px-2 py-2 text-sm font-medium text-gray-600 rounded-md hover:bg-gray-50 hover:text-gray-900 group">
                      <span className="mr-2">{item.icon}</span>
                      <span className="flex-1">{item.title}</span>
                      <ChevronRightIcon
                        className={`${
                          open ? 'transform rotate-90' : ''
                        } w-5 h-5 text-gray-400 transition-transform duration-150 ease-in-out group-hover:text-gray-500`}
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
                      <Disclosure.Panel className="space-y-1">
                        {item.items.map((subItem) => (
                          <Link
                            key={subItem.path}
                            href={subItem.path}
                            className={`group flex items-center pl-10 pr-2 py-2 text-sm font-medium rounded-md ${
                              router.asPath === subItem.path
                                ? 'text-primary-600 bg-primary-50'
                                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                            }`}
                          >
                            {subItem.title}
                          </Link>
                        ))}
                      </Disclosure.Panel>
                    </Transition>
                  </>
                )}
              </Disclosure>
            ) : (
              <Link
                href={item.path}
                className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                  router.asPath === item.path
                    ? 'text-primary-600 bg-primary-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.title}
              </Link>
            )}
          </div>
        ))}
      </div>
    </nav>
  )
} 
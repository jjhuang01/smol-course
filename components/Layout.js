import { useState, useEffect, Fragment } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { motion } from 'framer-motion'
import { Disclosure, Dialog, Transition } from '@headlessui/react'
import { ChevronRightIcon, Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

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
    ]
  },
  {
    title: '练习日志',
    icon: '📝',
    index: '4',
    items: [
      { title: '代码问题解决方案', path: '/docs/练习日志/解决方案/代码问题解决方案', index: '4.1' },
      { title: '模型问题解决方案', path: '/docs/练习日志/解决方案/模型问题解决方案', index: '4.2' },
      { title: '环境问题解决方案', path: '/docs/练习日志/解决方案/环境问题解决方案', index: '4.3' },
    ]
  }
]

function SidebarContent() {
  const router = useRouter()

  return (
    <nav className="mt-5 flex-1 px-2 space-y-1" aria-label="Sidebar">
      {menuItems.map((item) => (
        <div key={item.title} className="space-y-1">
          {item.items ? (
            <Disclosure defaultOpen={item.items.some(subItem => router.asPath === subItem.path)}>
              {({ open }) => (
                <>
                  <Disclosure.Button className="w-full flex items-center px-2 py-2 text-sm font-medium text-gray-600 rounded-md hover:bg-gray-50 hover:text-gray-900 group">
                    <div className="flex items-center min-w-[120px]">
                      <span className="mr-2">{item.icon}</span>
                      <span className="mr-2">{item.index}</span>
                      <span>{item.title}</span>
                    </div>
                    <ChevronRightIcon
                      className={`${
                        open ? 'transform rotate-90' : ''
                      } ml-auto w-5 h-5 text-gray-400 transition-transform duration-150 ease-in-out group-hover:text-gray-500`}
                    />
                  </Disclosure.Button>
                  <Disclosure.Panel className="space-y-1">
                    {item.items.map((subItem) => (
                      <Link
                        key={subItem.path}
                        href={subItem.path}
                        className={`group flex items-center pl-12 pr-2 py-2 text-sm font-medium rounded-md ${
                          router.asPath === subItem.path
                            ? 'text-primary-600 bg-primary-50'
                            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                        }`}
                      >
                        <div className="flex items-center min-w-[120px]">
                          <span className="mr-2">{subItem.index}</span>
                          <span>{subItem.title}</span>
                        </div>
                      </Link>
                    ))}
                  </Disclosure.Panel>
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
              <div className="flex items-center min-w-[120px]">
                <span className="mr-2">{item.icon}</span>
                <span className="mr-2">{item.index}</span>
                <span>{item.title}</span>
              </div>
            </Link>
          )}
        </div>
      ))}
    </nav>
  )
}

export default function Layout({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex h-screen overflow-hidden">
        {/* Mobile sidebar */}
        <div className="lg:hidden">
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
                <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white">
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
              <div className="flex-shrink-0 w-14">{/* Force sidebar to shrink to fit close icon */}</div>
            </Dialog>
          </Transition.Root>
        </div>

        {/* Desktop sidebar */}
        <div className="hidden lg:flex lg:flex-shrink-0">
          <div className="flex flex-col w-64 border-r border-gray-200 bg-white">
            <div className="flex-1 flex flex-col min-h-0">
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
          <div className="relative z-10 flex-shrink-0 flex h-16 bg-white shadow">
            <button
              type="button"
              className="px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 lg:hidden"
              onClick={() => setIsSidebarOpen(true)}
            >
              <span className="sr-only">打开侧边栏</span>
              <Bars3Icon className="h-6 w-6" aria-hidden="true" />
            </button>
          </div>

          <main className="flex-1 relative overflow-y-auto focus:outline-none">
            <div className="py-6">
              <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  {children}
                </motion.div>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
} 
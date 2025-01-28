import Link from 'next/link'
import { HomeIcon, ArrowLeftIcon } from '@heroicons/react/24/outline'

export default function Custom404() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 text-center">
        <div>
          <h1 className="text-6xl font-bold text-primary-600 mb-4">404</h1>
          <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">页面未找到</h2>
          <p className="mt-4 text-lg text-gray-500 dark:text-gray-400">
            抱歉，我们找不到您要访问的页面。这可能是因为：
          </p>
          <ul className="mt-4 text-left text-gray-500 dark:text-gray-400 space-y-2">
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>页面已被移动或删除</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>URL中可能存在拼写错误</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>您可能没有访问此页面的权限</span>
            </li>
          </ul>
        </div>
        <div className="mt-8 flex justify-center space-x-4">
          <Link 
            href="/"
            className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            <HomeIcon className="h-5 w-5 mr-2" />
            返回首页
          </Link>
          <button 
            onClick={() => window.history.back()}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-base font-medium rounded-md shadow-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            <ArrowLeftIcon className="h-5 w-5 mr-2" />
            返回上页
          </button>
        </div>
      </div>
    </div>
  )
} 
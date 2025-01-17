import Link from 'next/link'
import { motion } from 'framer-motion'

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
}

export default function Home() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial="hidden"
        animate="show"
        variants={container}
        className="max-w-4xl w-full space-y-8"
      >
        <motion.div variants={item} className="text-center">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-primary-600 to-primary-400 bg-clip-text text-transparent">
            SMOL Course
          </h1>
          <p className="mt-6 text-xl text-gray-600">
            欢迎来到 SMOL Course 学习平台
          </p>
        </motion.div>

        <motion.div
          variants={item}
          className="mt-12 grid grid-cols-1 gap-8 sm:grid-cols-2"
        >
          <Link
            href="/docs/学习指南"
            className="group relative rounded-xl border border-gray-200 p-6 bg-white shadow-sm transition-all duration-200 hover:shadow-lg hover:-translate-y-1"
          >
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <span className="text-3xl">📚</span>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600">
                  学习指南
                </h2>
                <p className="mt-2 text-gray-600">开始您的学习之旅</p>
              </div>
            </div>
          </Link>

          <Link
            href="/docs/项目说明"
            className="group relative rounded-xl border border-gray-200 p-6 bg-white shadow-sm transition-all duration-200 hover:shadow-lg hover:-translate-y-1"
          >
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <span className="text-3xl">📋</span>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600">
                  项目说明
                </h2>
                <p className="mt-2 text-gray-600">了解项目详情</p>
              </div>
            </div>
          </Link>

          <Link
            href="/docs/学习资料/AI关键词详解"
            className="group relative rounded-xl border border-gray-200 p-6 bg-white shadow-sm transition-all duration-200 hover:shadow-lg hover:-translate-y-1"
          >
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <span className="text-3xl">📖</span>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600">
                  学习资料
                </h2>
                <p className="mt-2 text-gray-600">浏览完整的学习资料</p>
              </div>
            </div>
          </Link>

          <Link
            href="/docs/练习日志/解决方案/代码问题解决方案"
            className="group relative rounded-xl border border-gray-200 p-6 bg-white shadow-sm transition-all duration-200 hover:shadow-lg hover:-translate-y-1"
          >
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <span className="text-3xl">📝</span>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600">
                  练习日志
                </h2>
                <p className="mt-2 text-gray-600">查看问题解决方案</p>
              </div>
            </div>
          </Link>
        </motion.div>
      </motion.div>
    </div>
  )
} 
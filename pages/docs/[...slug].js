import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import dynamic from 'next/dynamic'
import React, { useEffect } from 'react'
import Image from 'next/image'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'
import rehypePrism from 'rehype-prism-plus'
import remarkGfm from 'remark-gfm'
import { visit } from 'unist-util-visit'
import { all } from 'mdast-util-to-hast'

// 动态导入组件
const CodeBlock = dynamic(() => import('../../components/CodeBlock'), {
  loading: () => <div className="animate-pulse h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>,
  ssr: true
})

const Mermaid = dynamic(() => import('../../components/Mermaid'), {
  loading: () => <div className="animate-pulse h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>,
  ssr: true
})

// 加载状态组件
function LoadingState() {
  return (
    <div className="animate-pulse">
      <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
      <div className="space-y-3">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-4/6"></div>
      </div>
    </div>
  )
}

// 错误边界组件
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Content render error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 border border-red-500 rounded-lg">
          <h2 className="text-red-500 font-bold mb-2">内容加载出错</h2>
          <p>请刷新页面重试。如果问题持续存在，请联系支持团队。</p>
        </div>
      )
    }

    return this.props.children
  }
}

// 主文档组件
function Doc({ content = '', frontMatter, slug = [] }) {
  // 多层安全处理
  const safeContent = String(content || '')
    .replace(/\\</g, '&lt;')
    .replace(/\\>/g, '&gt;')

  // 改进标题提取逻辑
  const title = React.useMemo(() => {
    // 1. 优先使用 frontMatter 中的标题
    if (frontMatter?.title) {
      return frontMatter.title
    }

    // 2. 尝试从内容中提取 h1 标题
    const h1Match = safeContent.match(/^#\s+(.+?)(?:\n|$)/m)
    if (h1Match?.[1]) {
      return h1Match[1].trim()
    }

    // 3. 使用 slug 的最后一部分作为标题
    if (Array.isArray(slug) && slug.length > 0) {
      const lastSlug = decodeURIComponent(slug[slug.length - 1])
      return lastSlug.replace(/-/g, ' ').replace(/\.mdx?$/, '')
    }

    // 4. 不返回默认标题，而是返回空字符串
    return ''
  }, [frontMatter, safeContent, slug])

  // 添加调试信息
  useEffect(() => {
    const docPath = `docs/${Array.isArray(slug) ? slug.join('/') : slug}.md`
    console.log('Current document path:', docPath, '\nFormatted title:', title)
  }, [slug, title])

  // 处理加载状态
  if (!content && !frontMatter) {
    return <LoadingState />
  }

  // 移除内容中的第一个 h1 标题，因为我们已经在顶部显示了
  const contentWithoutH1 = safeContent.replace(/^#\s+(.+?)(?:\n|$)/m, '')

  return (
    <div className="prose dark:prose-invert max-w-none">
      {title && (
        <div className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm py-4 border-b border-gray-200 dark:border-gray-800">
          <h1 className="text-3xl font-bold mb-0">{title}</h1>
        </div>
      )}
      <div className="mdx-container">
        <ErrorBoundary>
          <React.Suspense fallback={<LoadingState />}>
            <div 
              className="mdx-content"
              dangerouslySetInnerHTML={{ __html: contentWithoutH1 }}
            />
          </React.Suspense>
        </ErrorBoundary>
      </div>
    </div>
  )
}

// 创建统一的Markdown处理器
const processor = unified()
  .use(remarkParse) // 解析Markdown
  .use(remarkGfm) // 支持GFM
  .use(() => (tree) => {
    visit(tree, 'table', (node) => {
      node.data = node.data || {}
      node.data.hProperties = { 
        className: 'markdown-table',
        'data-table': 'true'
      }
    })
    return tree
  })
  .use(remarkRehype, { 
    allowDangerousHtml: true,
    handlers: {
      table(h, node) {
        return h(node, 'table', { className: 'markdown-table' }, all(h, node))
      }
    }
  })
  .use(rehypePrism, {
    ignoreMissing: true,
    aliases: {
      prompt: 'bash'
    }
  })
  .use(rehypeStringify, { 
    allowDangerousHtml: true,
    closeSelfClosing: true
  })

// 辅助函数：规范化文件路径
function normalizePath(filePath) {
  return filePath
    .normalize('NFC')
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/') // 移除重复的斜杠
}

// 辅助函数：获取所有Markdown文件
export function getAllFiles(dirPath, arrayOfFiles = []) {
  const files = fs.readdirSync(dirPath, { withFileTypes: true })
  
  files.forEach(file => {
    const fullPath = normalizePath(path.join(dirPath, file.name))
    if (file.isDirectory()) {
      getAllFiles(fullPath, arrayOfFiles)
    } else if (/\.mdx?$/.test(file.name)) {
      arrayOfFiles.push(fullPath)
    }
  })

  return arrayOfFiles
}

// 路径别名映射
const PATH_ALIASES = {
  '深度学习架构详解': '深度学习架构'
}

export async function getStaticPaths() {
  try {
    const docsDirectory = path.join(process.cwd(), 'docs')
    const files = getAllFiles(docsDirectory)
    
    const paths = files.map(file => {
      const relativePath = path.relative(docsDirectory, file)
      const segments = relativePath
        .normalize('NFC') // 统一Unicode格式
        .replace(/\\/g, '/')
        .replace(/\.mdx?$/, '')
        .split('/')
        .filter(Boolean)
        .map(segment => decodeURIComponent(segment)) // 确保解码

      // 不再过滤中文字符，保持原始路径
      return segments.length > 0 ? { params: { slug: segments } } : null
    }).filter(Boolean)

    console.log('生成的有效路径示例:', paths.slice(0,3)) // 打印前3个路径示例
    
    return { paths, fallback: true }
  } catch (error) {
    console.error('路径生成失败:', error)
    return { paths: [], fallback: true }
  }
}

export async function getStaticProps({ params }) {
  try {
    // 确保params.slug存在
    if (!params?.slug) {
      return { notFound: true }
    }

    // 解码并规范化路径段，处理别名
    const slugSegments = params.slug.map(s => {
      const decoded = decodeURIComponent(s).normalize('NFC')
      return PATH_ALIASES[decoded] || decoded
    })
    
    // 尝试多种可能的文件路径
    const possiblePaths = [
      path.join(process.cwd(), 'docs', ...slugSegments) + '.md',
      path.join(process.cwd(), 'docs', ...slugSegments) + '.mdx',
      path.join(process.cwd(), 'docs', ...slugSegments, 'index.md'),
      path.join(process.cwd(), 'docs', ...slugSegments, 'index.mdx')
    ]

    console.log('尝试加载文件:', possiblePaths[0]) // 添加调试日志

    let filePath = null
    for (const possiblePath of possiblePaths) {
      if (fs.existsSync(possiblePath)) {
        filePath = possiblePath
        break
      }
    }

    if (!filePath) {
      console.warn('文件不存在，尝试路径:', possiblePaths)
      return { notFound: true }
    }

    const fileContent = fs.readFileSync(filePath, 'utf8')
    const { content, data } = matter(fileContent)

    // 添加更详细的内容验证
    if (!content || typeof content !== 'string') {
      console.error('无效的文件内容:', { filePath, content })
      return { notFound: true }
    }

    // 处理Markdown内容时添加错误边界
    let processedContent
    try {
      processedContent = String(await processor.process(content))
    } catch (processingError) {
      console.error('Markdown处理失败:', processingError)
      return { notFound: true }
    }

    return {
      props: {
        content: processedContent,
        frontMatter: data || {},
        slug: slugSegments.join('/'),
        filePath, // 用于调试
        params: JSON.parse(JSON.stringify(params)) // 确保可序列化
      },
      revalidate: 1, // 每秒重新验证一次（开发环境）
    }
  } catch (error) {
    console.error('生成文档失败:', error.message)
    return { notFound: true }
  }
}

export default Doc 
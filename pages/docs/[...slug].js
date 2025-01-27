import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { serialize } from 'next-mdx-remote/serialize'
import { MDXRemote } from 'next-mdx-remote'
import dynamic from 'next/dynamic'
import { memo } from 'react'
import CodeBlock from '../../components/CodeBlock'
import Mermaid from '../../components/Mermaid'
import Image from 'next/image'
import rehypePrism from 'rehype-prism-plus'
import remarkGfm from 'remark-gfm'
import { visit } from 'unist-util-visit'

// 动态导入Prism样式
const loadPrismStyles = () => {
  import('prismjs/themes/prism-tomorrow.css')
  import('prismjs/components/prism-python')
  import('prismjs/components/prism-bash')
  import('prismjs/components/prism-javascript')
  import('prismjs/components/prism-jsx')
  import('prismjs/components/prism-typescript')
  import('prismjs/components/prism-json')
  import('prismjs/components/prism-markdown')
}

// 在客户端动态加载样式
if (typeof window !== 'undefined') {
  loadPrismStyles()
}

// 优化样式对象
const proseStyles = {
  color: 'var(--tw-prose-body)',
  fontSize: '1.1rem',
  lineHeight: 1.7,
  backgroundColor: 'var(--tw-prose-bg)',
  padding: '2rem',
  borderRadius: '0.5rem',
  // 添加性能优化的CSS属性
  willChange: 'transform',
  backfaceVisibility: 'hidden',
}

const headingStyles = {
  h1: {
    color: 'var(--tw-prose-headings)',
    fontWeight: 800,
    fontSize: '2.5rem',
    marginBottom: '2rem',
    borderBottom: '2px solid var(--tw-prose-hr)',
    paddingBottom: '0.5rem',
    backgroundColor: 'var(--tw-prose-heading-bg)',
    padding: '1rem',
    borderRadius: '0.5rem',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  h2: {
    color: 'var(--tw-prose-headings)',
    fontWeight: 700,
    fontSize: '2rem',
    margin: '2rem 0 1rem',
    borderBottom: '1px solid var(--tw-prose-hr)',
    paddingBottom: '0.3rem',
    backgroundColor: 'var(--tw-prose-heading-bg)',
    padding: '0.75rem',
    borderRadius: '0.5rem',
  },
  h3: {
    color: 'var(--tw-prose-headings)',
    fontWeight: 600,
    fontSize: '1.5rem',
    margin: '1.5rem 0 1rem',
    backgroundColor: 'var(--tw-prose-heading-bg)',
    padding: '0.5rem',
    borderRadius: '0.5rem',
  },
}

// 动态导入重型组件
const DynamicMermaid = dynamic(() => import('../../components/Mermaid'), {
  loading: () => <div>Loading diagram...</div>,
  ssr: false
})

const DynamicCodeBlock = dynamic(() => import('../../components/CodeBlock'), {
  loading: () => <div>Loading code...</div>
})

// 使用memo优化组件重渲染
const components = {
  pre: memo((props) => <div {...props} />),
  code: DynamicCodeBlock,
  Image,
  Mermaid: DynamicMermaid
}

// 优化Doc组件
const Doc = memo(function Doc({ source, frontMatter }) {
  if (!source) {
    return <div>Loading...</div>
  }
  
  return (
    <div className="prose dark:prose-invert max-w-none">
      <MDXRemote {...source} components={components} />
    </div>
  )
})

export default Doc

export async function getStaticPaths() {
  const files = getAllFiles(path.join(process.cwd(), 'docs'))
  const paths = files.map(file => {
    const relativePath = path.relative(path.join(process.cwd(), 'docs'), file)
    const slug = relativePath.replace(/\.mdx?$/, '').split(path.sep)
    return {
      params: { slug }
    }
  })

  return {
    paths,
    fallback: false
  }
}

export async function getStaticProps({ params }) {
  try {
    const slug = params.slug.join('/')
    const filePath = path.join(process.cwd(), 'docs', `${slug}.md`)
    
    if (!fs.existsSync(filePath)) {
      return {
        notFound: true
      }
    }

    const source = fs.readFileSync(filePath, 'utf8')
    const { content, data } = matter(source)
    const mdxSource = await serialize(content, {
      mdxOptions: {
        remarkPlugins: [remarkGfm],
        rehypePlugins: [rehypePrism],
        development: process.env.NODE_ENV === 'development',
      },
      scope: data,
    })

    return {
      props: {
        source: mdxSource,
        frontMatter: data
      },
      revalidate: 60 * 60, // 1小时更新一次
    }
  } catch (error) {
    console.error('Error in getStaticProps:', error)
    return {
      notFound: true
    }
  }
}

function getAllFiles(dirPath, arrayOfFiles = []) {
  const cache = new Map()
  const cacheKey = dirPath

  if (cache.has(cacheKey)) {
    return cache.get(cacheKey)
  }

  const files = fs.readdirSync(dirPath)
  
  files.forEach(file => {
    const filePath = path.join(dirPath, file)
    if (fs.statSync(filePath).isDirectory()) {
      arrayOfFiles = getAllFiles(filePath, arrayOfFiles)
    } else if (file.endsWith('.md') || file.endsWith('.mdx')) {
      arrayOfFiles.push(filePath)
    }
  })

  cache.set(cacheKey, arrayOfFiles)
  return arrayOfFiles
} 
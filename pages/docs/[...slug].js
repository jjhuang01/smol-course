import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { serialize } from 'next-mdx-remote/serialize'
import { MDXRemote } from 'next-mdx-remote'
import CodeBlock from '../../components/CodeBlock'
import Mermaid from '../../components/Mermaid'
import Image from 'next/image'
import rehypePrism from 'rehype-prism-plus'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-jsx'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-markdown'
import remarkGfm from 'remark-gfm'
import { visit } from 'unist-util-visit'

const proseStyles = {
  color: 'var(--tw-prose-body)',
  fontSize: '1.1rem',
  lineHeight: 1.7,
  backgroundColor: 'var(--tw-prose-bg)',
  padding: '2rem',
  borderRadius: '0.5rem',
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

export default function Doc({ source, frontMatter }) {
  const components = {
    h1: (props) => <h1 style={headingStyles.h1} {...props} />,
    h2: (props) => <h2 style={headingStyles.h2} {...props} />,
    h3: (props) => <h3 style={headingStyles.h3} {...props} />,
    p: (props) => <p style={{ margin: '1.25em 0' }} {...props} />,
    ul: (props) => <ul style={{ margin: '1.25em 0', paddingLeft: '1.625em' }} {...props} />,
    li: (props) => <li style={{ margin: '0.5em 0' }} {...props} />,
    img: (props) => (
      <div style={{ position: 'relative', width: '100%', height: 'auto', margin: '2rem 0' }}>
        <Image
          {...props}
          width={800}
          height={450}
          style={{ maxWidth: '100%', height: 'auto' }}
          alt={props.alt || ''}
        />
      </div>
    ),
    code: CodeBlock,
    pre: (props) => {
      if (props.children?.props?.className?.includes('language-mermaid')) {
        let chart = props.children.props.children;
        
        // 调试信息
        console.log('Original chart content:', chart);
        
        // 提取 Mermaid 图表内容
        const extractMermaidContent = (node) => {
          if (typeof node === 'string') return node;
          if (Array.isArray(node)) {
            return node.map(item => {
              if (item.props?.children) {
                // 如果是带行号的代码行，提取实际内容
                const content = Array.isArray(item.props.children) 
                  ? item.props.children.map(child => 
                      typeof child === 'string' ? child : 
                      child.props?.children || ''
                    ).join('')
                  : item.props.children;
                return content;
              }
              return '';
            }).join('');
          }
          if (node?.props?.children) return extractMermaidContent(node.props.children);
          return '';
        };

        const chartContent = extractMermaidContent(chart)
          .replace(/\\n/g, '\n')
          .replace(/\\t/g, '\t')
          .replace(/\[object Object\]/g, '')
          .trim();
        
        console.log('Extracted chart content:', chartContent);
        
        // 验证图表内容
        if (!chartContent || chartContent.length === 0) {
          console.warn('Empty chart content detected');
          return null;
        }

        // 验证是否包含有效的 Mermaid 语法
        if (!chartContent.includes('sequenceDiagram') && 
            !chartContent.includes('graph') && 
            !chartContent.includes('gantt') && 
            !chartContent.includes('classDiagram') && 
            !chartContent.includes('stateDiagram')) {
          console.warn('Invalid Mermaid syntax detected:', chartContent);
          return (
            <div className="bg-red-50 dark:bg-red-900 p-4 rounded-lg">
              <p className="text-red-600 dark:text-red-200">无效的 Mermaid 图表语法</p>
              <pre className="mt-2 text-sm bg-white dark:bg-gray-800 p-2 rounded">{chartContent}</pre>
            </div>
          );
        }

        return (
          <div className="mermaid-wrapper">
            <Mermaid 
              chart={chartContent}
              config={{
                theme: 'dark',
                sequence: {
                  diagramMarginX: 50,
                  diagramMarginY: 10,
                  actorMargin: 100,
                  width: 150,
                  height: 65,
                  boxMargin: 10,
                  boxTextMargin: 5,
                  noteMargin: 10,
                  messageMargin: 35,
                  mirrorActors: false,
                  bottomMarginAdj: 1,
                  useMaxWidth: true,
                },
                themeVariables: {
                  sequenceNumberColor: '#60a5fa',
                  actorBorder: '#4b5563',
                  actorBkg: '#1f2937',
                  actorTextColor: '#e2e8f0',
                  actorLineColor: '#6b7280',
                  signalColor: '#60a5fa',
                  signalTextColor: '#e2e8f0',
                  labelBoxBkgColor: '#1f2937',
                  labelBoxBorderColor: '#4b5563',
                  labelTextColor: '#e2e8f0',
                  loopTextColor: '#e2e8f0',
                  noteBkgColor: '#374151',
                  noteBorderColor: '#4b5563',
                  noteTextColor: '#e2e8f0',
                  activationBkgColor: '#1f2937',
                  sequenceDiagramTitleColor: '#e2e8f0',
                }
              }}
            />
          </div>
        );
      }
      
      if (props.children?.props) {
        const { children, className } = props.children.props;
        
        let codeContent = '';
        const extractText = (node) => {
          if (typeof node === 'string') return node;
          if (Array.isArray(node)) return node.map(extractText).join('');
          if (node?.props?.children) return extractText(node.props.children);
          return '';
        };
        
        codeContent = extractText(children);
        
        codeContent = codeContent
          .replace(/\[object Object\]/g, '')
          .replace(/\\n/g, '\n')
          .replace(/\\t/g, '\t')
          .trim();
        
        return (
          <div className="code-block-wrapper">
            <CodeBlock className={className}>
              {codeContent}
            </CodeBlock>
          </div>
        );
      }
      
      return <pre {...props} />;
    },
  }

  return (
    <div className="prose-wrapper" style={{
      maxWidth: '65ch',
      margin: '0 auto',
      padding: '2rem',
    }}>
      <style jsx global>{`
        :root {
          --tw-prose-body: #374151;
          --tw-prose-headings: #111827;
          --tw-prose-hr: #e5e7eb;
          --tw-prose-bg: #ffffff;
          --tw-prose-heading-bg: #f3f4f6;
        }
        
        .dark {
          --tw-prose-body: #d1d5db;
          --tw-prose-headings: #f3f4f6;
          --tw-prose-hr: #374151;
          --tw-prose-bg: #1f2937;
          --tw-prose-heading-bg: #374151;
        }

        .prose-wrapper img {
          max-width: 100%;
          height: auto;
          border-radius: 0.5rem;
          margin: 2rem 0;
        }

        .prose-wrapper pre {
          background-color: #1e293b !important;
          color: #e2e8f0;
          padding: 1rem;
          border-radius: 0.5rem;
          overflow-x: auto;
          margin: 1.5rem 0;
        }

        .prose-wrapper code {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 0.875em;
        }

        /* Prism.js 语法高亮自定义样式 */
        .token.comment,
        .token.prolog,
        .token.doctype,
        .token.cdata {
          color: #8b9eb0;
        }

        .token.punctuation {
          color: #e2e8f0;
        }

        .token.property,
        .token.tag,
        .token.boolean,
        .token.number,
        .token.constant,
        .token.symbol,
        .token.deleted {
          color: #f687b3;
        }

        .token.selector,
        .token.attr-name,
        .token.string,
        .token.char,
        .token.builtin,
        .token.inserted {
          color: #84cc16;
        }

        .token.operator,
        .token.entity,
        .token.url,
        .language-css .token.string,
        .style .token.string {
          color: #a78bfa;
        }

        .token.atrule,
        .token.attr-value,
        .token.keyword {
          color: #60a5fa;
        }

        .token.function,
        .token.class-name {
          color: #f59e0b;
        }

        .token.regex,
        .token.important,
        .token.variable {
          color: #ec4899;
        }
      `}</style>
      <main>
        <article style={proseStyles}>
          <MDXRemote {...source} components={components} />
        </article>
      </main>
    </div>
  )
}

export async function getStaticPaths() {
  const files = getAllFiles('docs')
  const paths = files.map(file => ({
    params: {
      slug: file.replace(/\.md$/, '').split('/')
    }
  }))

  return {
    paths,
    fallback: false
  }
}

export async function getStaticProps({ params }) {
  const slug = params.slug.join('/')
  const filePath = path.join(process.cwd(), 'docs', `${slug}.md`)
  const source = fs.readFileSync(filePath, 'utf8')
  const { content, data } = matter(source)
  
  const mdxSource = await serialize(content, {
    mdxOptions: {
      remarkPlugins: [
        remarkGfm
      ],
      rehypePlugins: [
        [rehypePrism, { 
          showLineNumbers: true,
          ignoreMissing: true 
        }]
      ]
    },
    scope: data
  })

  return {
    props: {
      source: mdxSource,
      frontMatter: data
    }
  }
}

function getAllFiles(dirPath, arrayOfFiles) {
  const files = fs.readdirSync(dirPath)
  arrayOfFiles = arrayOfFiles || []

  files.forEach(file => {
    if (fs.statSync(path.join(dirPath, file)).isDirectory()) {
      arrayOfFiles = getAllFiles(path.join(dirPath, file), arrayOfFiles)
    } else {
      if (file.endsWith('.md')) {
        arrayOfFiles.push(path.join(dirPath, file).replace('docs/', ''))
      }
    }
  })

  return arrayOfFiles
} 
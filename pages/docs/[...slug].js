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
          <MDXRemote {...source} components={{
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
              const isMermaid = props.children?.props?.className === 'language-mermaid';
              if (isMermaid) {
                const chart = props.children.props.children;
                return (
                  <div className="my-8">
                    <Mermaid 
                      chart={chart}
                      config={{
                        theme: 'dark',
                        fontSize: 16,
                        useMaxWidth: false,
                        width: '100%',
                        height: '100%',
                        scale: 1.5
                      }}
                    />
                  </div>
                );
              }
              return <div className="code-block-wrapper"><CodeBlock {...props.children.props} /></div>;
            },
          }} />
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
      rehypePlugins: [
        [rehypePrism, { showLineNumbers: true }]
      ],
    },
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
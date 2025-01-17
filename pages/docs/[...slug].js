import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { serialize } from 'next-mdx-remote/serialize'
import { MDXRemote } from 'next-mdx-remote'
import CodeBlock from '../../components/CodeBlock'
import Mermaid from '../../components/Mermaid'

const proseStyles = {
  color: 'var(--tw-prose-body)',
  fontSize: '1.1rem',
  lineHeight: 1.7,
}

const headingStyles = {
  h1: {
    color: 'var(--tw-prose-headings)',
    fontWeight: 800,
    fontSize: '2.5rem',
    marginBottom: '2rem',
    borderBottom: '1px solid var(--tw-prose-hr)',
    paddingBottom: '0.5rem',
  },
  h2: {
    color: 'var(--tw-prose-headings)',
    fontWeight: 700,
    fontSize: '2rem',
    margin: '2rem 0 1rem',
    borderBottom: '1px solid var(--tw-prose-hr)',
    paddingBottom: '0.3rem',
  },
  h3: {
    color: 'var(--tw-prose-headings)',
    fontWeight: 600,
    fontSize: '1.5rem',
    margin: '1.5rem 0 1rem',
  },
}

export default function Doc({ source, frontMatter }) {
  return (
    <div style={{
      maxWidth: '65ch',
      margin: '0 auto',
      padding: '2rem',
    }}>
      <main>
        <article style={proseStyles}>
          <MDXRemote {...source} components={{
            h1: (props) => <h1 style={headingStyles.h1} {...props} />,
            h2: (props) => <h2 style={headingStyles.h2} {...props} />,
            h3: (props) => <h3 style={headingStyles.h3} {...props} />,
            p: (props) => <p style={{ margin: '1.25em 0' }} {...props} />,
            ul: (props) => <ul style={{ margin: '1.25em 0', paddingLeft: '1.625em' }} {...props} />,
            li: (props) => <li style={{ margin: '0.5em 0' }} {...props} />,
            code: CodeBlock,
            pre: (props) => {
              const isMermaid = props.children?.props?.className === 'language-mermaid';
              if (isMermaid) {
                return (
                  <div className="my-8">
                    <Mermaid 
                      chart={props.children.props.children}
                      config={{
                        theme: 'default',
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
              return <CodeBlock {...props.children.props} />;
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
  const mdxSource = await serialize(content)

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
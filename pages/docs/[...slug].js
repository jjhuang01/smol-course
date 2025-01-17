import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { serialize } from 'next-mdx-remote/serialize'
import { MDXRemote } from 'next-mdx-remote'

export default function Doc({ source, frontMatter }) {
  return (
    <div className="container">
      <main>
        <article className="prose">
          <MDXRemote {...source} />
        </article>
      </main>

      <style jsx>{`
        .container {
          max-width: 65ch;
          margin: 0 auto;
          padding: 2rem;
        }

        .prose {
          font-size: 1.1rem;
          line-height: 1.7;
        }

        .prose h1 {
          font-size: 2.5rem;
          margin-bottom: 2rem;
        }

        .prose h2 {
          font-size: 2rem;
          margin: 2rem 0 1rem;
        }

        .prose h3 {
          font-size: 1.5rem;
          margin: 1.5rem 0 1rem;
        }
      `}</style>
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
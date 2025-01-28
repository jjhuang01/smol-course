import { getAllFiles } from './docs/[...slug]'
import path from 'path'

export default function Debug({ files }) {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">调试信息</h1>
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">可用文件:</h2>
        <ul className="list-disc pl-5 space-y-2">
          {files.map((file, index) => (
            <li key={index} className="text-sm font-mono">
              {file}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export async function getStaticProps() {
  try {
    const docsDirectory = path.join(process.cwd(), 'docs')
    const files = getAllFiles(docsDirectory)
    
    return {
      props: {
        files
      }
    }
  } catch (error) {
    console.error('Error in debug page:', error)
    return {
      props: {
        files: []
      }
    }
  }
} 
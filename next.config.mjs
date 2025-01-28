import createMDX from '@next/mdx';
import rehypePrism from 'rehype-prism-plus';
import remarkGfm from 'remark-gfm';

const withMDX = createMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [remarkGfm],
    rehypePlugins: [
      [rehypePrism, { 
        showLineNumbers: true, 
        ignoreMissing: true,
        aliases: {
          text: 'plaintext',
          sh: 'bash',
          js: 'javascript',
          prompt: 'bash'
        }
      }]
    ],
  },
})

/** @type {import('next').NextConfig} */
const nextConfig = {
  // 启用 MDX 支持
  pageExtensions: ['js', 'jsx', 'md', 'mdx'],
  
  // 配置图片优化
  images: {
    unoptimized: true
  },

  // 确保 Mermaid 组件在客户端渲染
  webpack: (config) => {
    // 添加 mermaid 支持
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
    };
    
    // 支持更多语言的语法高亮
    config.module.rules.push({
      test: /\.mjs$/,
      include: /node_modules/,
      type: 'javascript/auto',
    });
    
    return config;
  },

  async rewrites() {
    return [
      {
        source: '/_next/data/:path*',
        destination: '/_next/data/:path*',
        has: [
          {
            type: 'query',
            key: 'slug',
            value: '(?<slug>.*)'
          }
        ]
      }
    ]
  }
}

export default withMDX(nextConfig) 
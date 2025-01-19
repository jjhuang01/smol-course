import createMDX from '@next/mdx';

const withMDX = createMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [],
    rehypePlugins: [
      ['rehype-prism-plus', { showLineNumbers: true }]
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
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
    };
    return config;
  },
}

export default withMDX(nextConfig) 
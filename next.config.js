const withMDX = require('@next/mdx')({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [],
    rehypePlugins: [],
  },
})

/** @type {import('next').NextConfig} */
const nextConfig = {
  // 启用 MDX 支持
  pageExtensions: ['js', 'jsx', 'md', 'mdx'],
  
  // 配置图片优化
  images: {
    unoptimized: true
  }
}

module.exports = withMDX(nextConfig) 
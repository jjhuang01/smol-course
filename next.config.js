/** @type {import('next').NextConfig} */
const nextConfig = {
  // 启用 MDX 支持
  pageExtensions: ['js', 'jsx', 'md', 'mdx'],

  // 配置图片域名
  images: {
    unoptimized: true
  }
}

module.exports = nextConfig 
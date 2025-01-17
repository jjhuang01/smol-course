module.exports = {
  // 启用 MDX 支持
  pageExtensions: ['js', 'jsx', 'md', 'mdx'],
  
  // 配置国际化
  i18n: {
    locales: ['default', 'es', 'ja', 'ko', 'pt-br', 'vi'],
    defaultLocale: 'default'
  },

  // 配置图片域名
  images: {
    domains: ['localhost'],
    unoptimized: true
  }
} 
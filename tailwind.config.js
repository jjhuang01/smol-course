/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,md,mdx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.gray.900'),
            maxWidth: 'none',
            p: {
              marginTop: '1.5em',
              marginBottom: '1.5em',
              lineHeight: '1.75',
            },
            'h1, h2, h3, h4': {
              marginTop: '2em',
              marginBottom: '1em',
              lineHeight: '1.3',
            },
            h1: {
              fontSize: '2.5em',
            },
            h2: {
              fontSize: '2em',
            },
            h3: {
              fontSize: '1.5em',
            },
            'ul, ol': {
              marginTop: '1.5em',
              marginBottom: '1.5em',
              paddingLeft: '1.5em',
            },
            li: {
              marginTop: '0.5em',
              marginBottom: '0.5em',
            },
            a: {
              color: theme('colors.primary.600'),
              textDecoration: 'none',
              '&:hover': {
                color: theme('colors.primary.700'),
              },
            },
            pre: {
              marginTop: '1.5em',
              marginBottom: '1.5em',
              padding: '1.5em',
              borderRadius: '0.5em',
            },
            code: {
              fontSize: '0.875em',
              fontWeight: '600',
            },
            img: {
              marginTop: '2em',
              marginBottom: '2em',
              borderRadius: '0.5em',
            },
            blockquote: {
              marginTop: '1.5em',
              marginBottom: '1.5em',
              paddingLeft: '1.5em',
            },
          },
        },
        dark: {
          css: {
            color: theme('colors.gray.200'),
            a: {
              color: theme('colors.primary.400'),
              '&:hover': {
                color: theme('colors.primary.300'),
              },
            },
            'h1, h2, h3, h4': {
              color: theme('colors.gray.100'),
            },
            p: {
              color: theme('colors.gray.300'),
            },
            strong: {
              color: theme('colors.gray.100'),
            },
            blockquote: {
              color: theme('colors.gray.300'),
              borderLeftColor: theme('colors.gray.600'),
            },
            code: {
              color: theme('colors.gray.200'),
            },
            pre: {
              backgroundColor: theme('colors.gray.900'),
            },
            'ul, ol': {
              color: theme('colors.gray.300'),
            },
            li: {
              color: theme('colors.gray.300'),
            },
            hr: {
              borderColor: theme('colors.gray.700'),
            },
            table: {
              th: {
                color: theme('colors.gray.200'),
              },
              td: {
                color: theme('colors.gray.300'),
              },
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}


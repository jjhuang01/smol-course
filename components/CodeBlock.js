import { useState, useEffect } from 'react';
import { ClipboardIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline';
import Prism from 'prismjs';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-mermaid';
import 'prismjs/plugins/line-numbers/prism-line-numbers';
import 'prismjs/themes/prism-tomorrow.css';
import 'prismjs/plugins/line-numbers/prism-line-numbers.css';

export default function CodeBlock({ children, className }) {
  const [copied, setCopied] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    if (typeof window !== 'undefined') {
      Prism.highlightAll();
    }
  }, [children]);

  const handleCopy = async () => {
    const code = children.props ? children.props.children : children;
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // 获取代码内容
  const getCodeContent = () => {
    if (!children) return '';
    
    // 递归提取文本内容
    const extractText = (node) => {
      if (typeof node === 'string') return node;
      if (Array.isArray(node)) return node.map(extractText).join('');
      if (node?.props?.children) return extractText(node.props.children);
      return '';
    };
    
    let content = extractText(children);
    
    // 处理特殊字符
    content = content
      .replace(/\[object Object\]/g, '')
      .replace(/\\n/g, '\n')
      .replace(/\\t/g, '\t')
      .trim();
    
    return content;
  };

  // 获取语言类型
  const getLanguage = () => {
    if (className) {
      const match = className.match(/language-(\w+)/);
      return match ? match[1] : '';
    }
    return '';
  };

  // 确保类名顺序一致
  const getClassName = (baseClass) => {
    const defaultClasses = ['code-highlight', 'relative', 'line-numbers'];
    const language = getLanguage();
    if (language) {
      defaultClasses.push(`language-${language}`);
    }
    if (!baseClass) return defaultClasses.join(' ');
    
    const classes = baseClass.split(' ').filter(Boolean);
    const uniqueClasses = [...new Set([...defaultClasses, ...classes])];
    return uniqueClasses.join(' ');
  };

  const codeClassName = getClassName(className);
  const content = getCodeContent();

  return (
    <div className="relative group">
      {mounted && (
        <button
          onClick={handleCopy}
          className="absolute right-2 top-2 p-2 rounded-lg bg-gray-800 dark:bg-gray-700 text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          title={copied ? "已复制！" : "复制代码"}
        >
          {copied ? (
            <ClipboardDocumentCheckIcon className="h-5 w-5 text-green-400" />
          ) : (
            <ClipboardIcon className="h-5 w-5" />
          )}
        </button>
      )}
      <pre 
        className={codeClassName}
        style={{
          background: '#1e293b',
          padding: '1rem',
          borderRadius: '0.5rem',
          overflow: 'auto',
          margin: '1.5rem 0'
        }}
        suppressHydrationWarning
      >
        <code className={codeClassName} suppressHydrationWarning>
          {content}
        </code>
      </pre>
      <style jsx global>{`
        pre[class*="language-"] {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 0.875em;
          line-height: 1.7;
          direction: ltr;
          text-align: left;
          white-space: pre;
          word-spacing: normal;
          word-break: normal;
          tab-size: 4;
          hyphens: none;
          color: #e2e8f0;
          background: #1e293b;
          border-radius: 0.5rem;
          padding: 1em;
          margin: 0.5em 0;
          overflow: auto;
        }

        code[class*="language-"] {
          font-family: inherit;
          font-size: inherit;
          line-height: inherit;
          tab-size: 4;
          white-space: pre-wrap;
          word-break: break-word;
        }

        /* Line Numbers */
        pre[class*="language-"].line-numbers {
          position: relative;
          padding-left: 3.8em;
          counter-reset: linenumber;
        }

        pre[class*="language-"].line-numbers > code {
          position: relative;
          white-space: inherit;
        }

        .line-numbers .line-numbers-rows {
          position: absolute;
          pointer-events: none;
          top: 0;
          font-size: 100%;
          left: -3.8em;
          width: 3em;
          letter-spacing: -1px;
          border-right: 1px solid #64748b;
          user-select: none;
        }

        .line-numbers-rows > span {
          display: block;
          counter-increment: linenumber;
          pointer-events: none;
        }

        .line-numbers-rows > span:before {
          content: counter(linenumber);
          color: #64748b;
          display: block;
          padding-right: 0.8em;
          text-align: right;
        }

        /* Token colors */
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

        .token.important,
        .token.bold {
          font-weight: bold;
        }

        .token.italic {
          font-style: italic;
        }

        .token.entity {
          cursor: help;
        }

        /* Mermaid 图表样式 */
        .language-mermaid {
          background: transparent !important;
        }

        .language-mermaid svg {
          max-width: 100%;
          height: auto;
        }
      `}</style>
    </div>
  );
} 
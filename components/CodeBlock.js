import { useState } from 'react';
import { ClipboardIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline';

export default function CodeBlock({ children, className }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const code = children.props ? children.props.children : children;
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group">
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
      <pre className={`${className} relative`} style={{
        background: '#1e293b',
        padding: '1rem',
        borderRadius: '0.5rem',
        overflow: 'auto',
      }}>
        <code className={className}>
          {children}
        </code>
      </pre>
      <style jsx global>{`
        pre {
          margin: 1.5rem 0;
        }
        
        pre code {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 0.875em;
          line-height: 1.7;
        }

        .line-number {
          display: inline-block;
          width: 2.5em;
          color: #64748b;
          text-align: right;
          padding-right: 1em;
          user-select: none;
        }

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
      `}</style>
    </div>
  );
} 
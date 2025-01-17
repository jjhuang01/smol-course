import { useState } from 'react';
import { ClipboardIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline';

export default function CodeBlock({ children, className }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const code = children.props.children;
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
      <pre className={className}>
        {children}
      </pre>
    </div>
  );
} 
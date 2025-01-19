import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';

function MermaidComponent({ chart, config = {} }) {
  const ref = useRef(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    // 动态导入并初始化 mermaid
    import('mermaid').then((mermaid) => {
      mermaid.default.initialize({
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose',
        themeVariables: {
          fontFamily: 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif',
        }
      });
    });
  }, []);

  useEffect(() => {
    if (isClient && ref.current) {
      import('mermaid').then((mermaid) => {
        try {
          mermaid.default.contentLoading = false;
          ref.current.innerHTML = chart;
          mermaid.default.init(config, ref.current);
        } catch (error) {
          console.error('Mermaid initialization failed:', error);
        }
      });
    }
  }, [chart, config, isClient]);

  // 服务端渲染时返回一个占位符
  if (!isClient) {
    return <div className="mermaid-placeholder" />;
  }

  return (
    <div className="mermaid-wrapper my-4">
      <div className="mermaid" ref={ref} />
    </div>
  );
}

// 使用 dynamic 导入确保组件只在客户端渲染
export default dynamic(() => Promise.resolve(MermaidComponent), {
  ssr: false
}); 
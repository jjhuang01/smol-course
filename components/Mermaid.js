import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';

function MermaidComponent({ chart, config = {} }) {
  const ref = useRef(null);
  const [isClient, setIsClient] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient || !ref.current || !chart) return;

    const renderChart = async () => {
      try {
        const mermaid = (await import('mermaid')).default;
        
        // 调试信息
        console.log('Mermaid chart content:', chart);
        
        // 验证图表内容
        if (!chart || chart.trim().length === 0) {
          throw new Error('Empty chart content');
        }

        // 初始化配置
        mermaid.initialize({
          startOnLoad: false,
          theme: 'dark',
          logLevel: 'debug', // 设置为 debug 以获取更多信息
          securityLevel: 'loose',
          sequence: {
            diagramMarginX: 50,
            diagramMarginY: 10,
            actorMargin: 100,
            width: 150,
            height: 65,
            boxMargin: 10,
            boxTextMargin: 5,
            noteMargin: 10,
            messageMargin: 35,
            mirrorActors: false,
            bottomMarginAdj: 1,
            useMaxWidth: true,
          },
          themeVariables: {
            sequenceNumberColor: '#60a5fa',
            actorBorder: '#4b5563',
            actorBkg: '#1f2937',
            actorTextColor: '#e2e8f0',
            actorLineColor: '#6b7280',
            signalColor: '#60a5fa',
            signalTextColor: '#e2e8f0',
            labelBoxBkgColor: '#1f2937',
            labelBoxBorderColor: '#4b5563',
            labelTextColor: '#e2e8f0',
            loopTextColor: '#e2e8f0',
            noteBkgColor: '#374151',
            noteBorderColor: '#4b5563',
            noteTextColor: '#e2e8f0',
            activationBkgColor: '#1f2937',
            sequenceDiagramTitleColor: '#e2e8f0',
          },
          ...config
        });

        // 清除之前的内容
        ref.current.innerHTML = '';
        
        // 生成唯一的图表 ID
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        
        // 解析和验证图表
        const graphDefinition = chart.trim();
        console.log('Parsing graph definition:', graphDefinition);
        
        // 先验证语法
        const valid = await mermaid.parse(graphDefinition);
        console.log('Graph validation result:', valid);
        
        if (!valid) {
          throw new Error('Invalid graph syntax');
        }
        
        // 渲染图表
        const { svg } = await mermaid.render(id, graphDefinition);
        console.log('Generated SVG:', svg.substring(0, 100) + '...');
        
        // 设置新的 SVG
        ref.current.innerHTML = svg;
        
        // 重置错误状态
        setError(null);
      } catch (err) {
        console.error('Mermaid rendering error:', err);
        setError(err);
        
        // 在渲染失败时显示原始代码和错误信息
        ref.current.innerHTML = `
          <div class="bg-red-50 dark:bg-red-900 p-4 rounded-lg">
            <p class="text-red-600 dark:text-red-200">图表渲染失败</p>
            <pre class="mt-2 text-sm bg-white dark:bg-gray-800 p-2 rounded">${err.message}</pre>
            <div class="mt-4 bg-gray-50 dark:bg-gray-900 p-4 rounded">
              <code class="text-sm">${chart}</code>
            </div>
          </div>
        `;
      }
    };

    renderChart();
  }, [chart, config, isClient]);

  if (!isClient) {
    return null;
  }

  return (
    <div className="mermaid-wrapper">
      <div ref={ref} className="mermaid dark:text-white" />
    </div>
  );
}

export default dynamic(() => Promise.resolve(MermaidComponent), {
  ssr: false
}); 
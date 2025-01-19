import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { ArrowsPointingOutIcon, ArrowsPointingInIcon, XMarkIcon } from '@heroicons/react/24/outline';

function MermaidComponent({ chart, config = {} }) {
  const ref = useRef(null);
  const [isClient, setIsClient] = useState(false);
  const [error, setError] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [scale, setScale] = useState(1);
  const fullscreenRef = useRef(null);

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

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    setScale(1); // 重置缩放
  };

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5));
  };

  if (!isClient) {
    return null;
  }

  return (
    <>
      <div className="mermaid-wrapper group relative">
        <div ref={ref} className="mermaid dark:text-white" />
        <button
          onClick={toggleFullscreen}
          className="absolute top-2 right-2 p-2 bg-gray-800/70 hover:bg-gray-800 rounded-lg text-white opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          title="全屏查看"
        >
          <ArrowsPointingOutIcon className="h-5 w-5" />
        </button>
      </div>

      {isFullscreen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="relative w-full h-full max-w-[90vw] max-h-[90vh] m-4 bg-white dark:bg-gray-800 rounded-lg shadow-xl overflow-auto">
            <div className="sticky top-0 z-10 flex justify-between items-center p-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleZoomOut}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                  title="缩小"
                >
                  <ArrowsPointingInIcon className="h-5 w-5" />
                </button>
                <span className="text-sm text-gray-600 dark:text-gray-300">
                  {Math.round(scale * 100)}%
                </span>
                <button
                  onClick={handleZoomIn}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                  title="放大"
                >
                  <ArrowsPointingOutIcon className="h-5 w-5" />
                </button>
              </div>
              <button
                onClick={toggleFullscreen}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                title="关闭全屏"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>
            <div 
              ref={fullscreenRef}
              className="p-4 flex items-center justify-center min-h-[calc(100vh-10rem)]"
              style={{
                transform: `scale(${scale})`,
                transformOrigin: 'center center',
                transition: 'transform 0.2s ease-in-out'
              }}
            >
              <div className="mermaid dark:text-white" dangerouslySetInnerHTML={{ __html: ref.current?.innerHTML || '' }} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default dynamic(() => Promise.resolve(MermaidComponent), {
  ssr: false
}); 
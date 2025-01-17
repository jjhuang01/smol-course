import { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  themeVariables: {
    fontFamily: 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif',
  }
});

export default function Mermaid({ chart }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      mermaid.contentLoading = false;
      mermaid.init(undefined, ref.current);
    }
  }, [chart]);

  return (
    <div className="mermaid-wrapper my-4">
      <div className="mermaid" ref={ref}>
        {chart}
      </div>
    </div>
  );
} 
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

export default function TableOfContents() {
  const [headings, setHeadings] = useState([]);
  const [activeId, setActiveId] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);
  const [isMounted, setIsMounted] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  // 监听路由变化，重新扫描标题
  useEffect(() => {
    if (!isMounted) return;
    
    const scanHeadings = () => {
      try {
        const idCounts = new Map(); // 用于跟踪 ID 的使用次数
        const elements = Array.from(document.querySelectorAll('h2, h3, h4') || [])
          .filter(element => element && element.textContent)
          .map(element => {
            const text = element.textContent.trim();
            let id = text
              .toLowerCase()
              .replace(/\s+/g, '-')
              .replace(/[^a-z0-9-\u4e00-\u9fa5]/g, '');
            
            // 如果 ID 已存在，添加计数后缀
            if (idCounts.has(id)) {
              const count = idCounts.get(id) + 1;
              idCounts.set(id, count);
              id = `${id}-${count}`;
            } else {
              idCounts.set(id, 1);
            }
            
            element.id = element.id || id;
            
            return {
              id: element.id,
              text,
              level: Number(element.tagName.charAt(1)),
            };
          })
          .filter(heading => heading.id && heading.text);
        
        setHeadings(elements);
        setupIntersectionObserver(elements);
      } catch (error) {
        console.error('Error scanning headings:', error);
      }
    };

    // 等待 DOM 更新完成后再扫描标题
    const timer = setTimeout(scanHeadings, 100);
    return () => clearTimeout(timer);
  }, [isMounted, router.asPath]); // 添加 router.asPath 作为依赖

  const setupIntersectionObserver = (elements) => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting && entry.target.id) {
            setActiveId(entry.target.id);
          }
        });
      },
      { 
        rootMargin: '-20% 0% -35% 0%',
        threshold: 0.5
      }
    );

    elements.forEach(heading => {
      const element = document.getElementById(heading.id);
      if (element) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  };

  const scrollToHeading = (id) => {
    try {
      const element = document.getElementById(id);
      if (element) {
        // 获取固定导航栏的高度
        const navHeight = document.querySelector('nav')?.offsetHeight || 0;
        const offset = navHeight + 20; // 额外的偏移量

        // 使用 scrollIntoView 实现平滑滚动
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        // 补偿固定导航栏的高度
        setTimeout(() => {
          window.scrollBy({
            top: -offset,
            behavior: 'smooth'
          });
        }, 100);

        setActiveId(id);
      }
    } catch (error) {
      console.error('Error scrolling to heading:', error);
    }
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // 监听滚动显示/隐藏返回顶部按钮
  const [showBackToTop, setShowBackToTop] = useState(false);
  useEffect(() => {
    if (!isMounted) return;

    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 300);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [isMounted]);

  // 在移动端不显示目录
  if (typeof window !== 'undefined' && window.innerWidth < 1280) {
    return null;
  }

  if (!isMounted || !headings || headings.length === 0) return null;

  return (
    <>
      <nav className="hidden xl:block fixed right-8 top-24 w-64 overflow-y-auto max-h-[calc(100vh-8rem)]">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">目录</h2>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors duration-200"
              aria-label={isExpanded ? "收起目录" : "展开目录"}
            >
              {isExpanded ? (
                <ChevronUpIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
              ) : (
                <ChevronDownIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
              )}
            </button>
          </div>
          {isExpanded && headings.length > 0 && (
            <ul className="space-y-2">
              {headings.map(heading => (
                <li
                  key={heading.id}
                  style={{ paddingLeft: `${(heading.level - 2) * 1}rem` }}
                >
                  <button
                    onClick={() => scrollToHeading(heading.id)}
                    className={`w-full text-left py-1 text-sm transition-colors duration-200 ${
                      activeId === heading.id
                        ? 'text-primary-600 dark:text-primary-400 font-medium'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                    }`}
                  >
                    {heading.text}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </nav>

      {showBackToTop && (
        <button
          onClick={scrollToTop}
          className="fixed right-8 bottom-8 p-2 bg-primary-500 text-white rounded-full shadow-lg hover:bg-primary-600 transition-colors duration-200"
          aria-label="返回顶部"
        >
          <ChevronUpIcon className="h-6 w-6" />
        </button>
      )}
    </>
  );
} 
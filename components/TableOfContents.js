import { useEffect, useState } from 'react';
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

export default function TableOfContents() {
  const [headings, setHeadings] = useState([]);
  const [activeId, setActiveId] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll('h2, h3, h4'))
      .filter(element => element && element.textContent)
      .map(element => {
        const text = element.textContent.trim();
        const id = text.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-');
        // 设置元素的 ID
        if (!element.id) {
          element.id = id;
        }
        return {
          id: element.id || id,
          text,
          level: Number(element.tagName.charAt(1)),
        };
      })
      .filter(heading => heading.id && heading.text);
    
    setHeadings(elements);

    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting && entry.target.id) {
            setActiveId(entry.target.id);
            // 更新 URL hash，但不触发滚动
            const newUrl = window.location.pathname + '#' + entry.target.id;
            window.history.replaceState(null, '', newUrl);
          }
        });
      },
      { rootMargin: '-20% 0% -35% 0%' }
    );

    elements.forEach(heading => {
      const element = document.getElementById(heading.id);
      if (element) {
        observer.observe(element);
      }
    });

    return () => {
      elements.forEach(heading => {
        const element = document.getElementById(heading.id);
        if (element) {
          observer.unobserve(element);
        }
      });
    };
  }, []);

  const [showBackToTop, setShowBackToTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 300);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleClick = (e, id) => {
    e.preventDefault();
    const element = document.getElementById(id);
    if (element) {
      const offset = 80; // 顶部偏移量，避免被固定导航栏遮挡
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - offset;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });

      // 更新 URL，但不触发默认的滚动行为
      window.history.pushState(null, '', `#${id}`);
      setActiveId(id);
    }
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (!headings || headings.length === 0) return null;

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
              {headings.map(heading => heading && (
                <li
                  key={heading.id}
                  style={{ paddingLeft: `${(heading.level - 2) * 1}rem` }}
                >
                  <a
                    href={`#${heading.id}`}
                    onClick={(e) => handleClick(e, heading.id)}
                    className={`block py-1 text-sm transition-colors duration-200 ${
                      activeId === heading.id
                        ? 'text-primary-600 dark:text-primary-400 font-medium'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                    }`}
                  >
                    {heading.text}
                  </a>
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
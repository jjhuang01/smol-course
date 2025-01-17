import { useEffect, useState } from 'react';
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

export default function TableOfContents() {
  const [headings, setHeadings] = useState([]);
  const [activeId, setActiveId] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll('h2, h3, h4'))
      .map(element => ({
        id: element.id || element.textContent.toLowerCase().replace(/\s+/g, '-'),
        text: element.textContent,
        level: Number(element.tagName.charAt(1)),
      }));
    
    setHeadings(elements);

    // 为没有 id 的标题添加 id
    elements.forEach(heading => {
      if (!document.getElementById(heading.id)) {
        const element = Array.from(document.querySelectorAll(`h${heading.level}`))
          .find(el => el.textContent === heading.text);
        if (element) {
          element.id = heading.id;
        }
      }
    });

    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      { rootMargin: '0% 0% -80% 0%' }
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

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (headings.length === 0) return null;

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
          {isExpanded && (
            <ul className="space-y-2">
              {headings.map(heading => (
                <li
                  key={heading.id}
                  style={{ paddingLeft: `${(heading.level - 2) * 1}rem` }}
                >
                  <a
                    href={`#${heading.id}`}
                    className={`block py-1 text-sm transition-colors duration-200 ${
                      activeId === heading.id
                        ? 'text-primary-600 dark:text-primary-400'
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
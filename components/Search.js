import { useState, useEffect, useRef } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/router';
import Link from 'next/link';

export default function Search({ menuItems }) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);
  const searchRef = useRef(null);
  const router = useRouter();

  // 扁平化菜单项
  const flattenMenuItems = (items) => {
    let flattened = [];
    items.forEach(item => {
      if (item.path) {
        flattened.push({
          title: item.title,
          path: item.path,
          index: item.index
        });
      }
      if (item.items) {
        flattened = [...flattened, ...flattenMenuItems(item.items)];
      }
    });
    return flattened;
  };

  // 搜索逻辑
  const handleSearch = (term) => {
    setSearchTerm(term);
    if (!term.trim()) {
      setResults([]);
      return;
    }

    const flatItems = flattenMenuItems(menuItems);
    const searchResults = flatItems.filter(item =>
      item.title.toLowerCase().includes(term.toLowerCase())
    );
    setResults(searchResults);
  };

  // 点击外部关闭搜索
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={searchRef}>
      <div className="flex items-center">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="搜索"
        >
          <MagnifyingGlassIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        </button>
      </div>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-72 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="p-2">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="搜索文档..."
              className="w-full px-3 py-2 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
              autoFocus
            />
          </div>
          
          {results.length > 0 && (
            <ul className="max-h-60 overflow-y-auto py-2">
              {results.map((result) => (
                <li key={result.path}>
                  <Link
                    href={result.path}
                    className="flex items-center px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
                    onClick={() => setIsOpen(false)}
                  >
                    <span className="text-sm text-gray-600 dark:text-gray-300 mr-2">
                      {result.index}
                    </span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {result.title}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}

          {searchTerm && results.length === 0 && (
            <div className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400">
              未找到相关内容
            </div>
          )}
        </div>
      )}
    </div>
  );
} 
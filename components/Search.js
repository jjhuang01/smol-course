import { useState, useEffect, useRef } from 'react';
import { MagnifyingGlassIcon, CommandIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/router';
import Link from 'next/link';

export default function Search({ menuItems }) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);
  const searchRef = useRef(null);
  const inputRef = useRef(null);
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
      item.title.toLowerCase().includes(term.toLowerCase()) ||
      item.index.toLowerCase().includes(term.toLowerCase())
    );
    setResults(searchResults);
  };

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl/Cmd + K 打开搜索
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
      // Esc 关闭搜索
      if (e.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

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

  // 打开搜索时自动聚焦输入框
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  return (
    <div className="relative" ref={searchRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center space-x-2"
        aria-label="搜索"
      >
        <MagnifyingGlassIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        <div className="hidden sm:flex items-center text-sm text-gray-500 dark:text-gray-400">
          <span className="mr-2">搜索</span>
          <kbd className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded-md">
            <span className="mr-1">⌘</span>K
          </kbd>
        </div>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="p-2">
            <div className="relative">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400 absolute left-3 top-2.5" />
              <input
                ref={inputRef}
                type="text"
                value={searchTerm}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="搜索文档..."
                className="w-full pl-10 pr-3 py-2 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 px-3">
              搜索提示：输入编号(如 3.1)或关键词
            </div>
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
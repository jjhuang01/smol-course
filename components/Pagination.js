import Link from 'next/link';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

export default function Pagination({ prevPage, nextPage }) {
  return (
    <nav className="mt-8 flex justify-between border-t border-gray-200 dark:border-gray-700 pt-6">
      <div className="flex-1 min-w-0">
        {prevPage ? (
          <Link
            href={prevPage.path}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <ChevronLeftIcon className="h-5 w-5 mr-1 flex-shrink-0" />
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-wide">上一篇</div>
              <div className="mt-1 text-base font-medium truncate">{prevPage.title}</div>
            </div>
          </Link>
        ) : (
          <div className="invisible">占位</div>
        )}
      </div>
      
      <div className="flex-1 min-w-0 flex justify-end">
        {nextPage ? (
          <Link
            href={nextPage.path}
            className="flex items-center text-right text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-wide">下一篇</div>
              <div className="mt-1 text-base font-medium truncate">{nextPage.title}</div>
            </div>
            <ChevronRightIcon className="h-5 w-5 ml-1 flex-shrink-0" />
          </Link>
        ) : (
          <div className="invisible">占位</div>
        )}
      </div>
    </nav>
  );
} 
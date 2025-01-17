import Link from 'next/link';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

export default function Pagination({ prevPage, nextPage }) {
  return (
    <div className="mt-8 flex justify-between border-t border-gray-200 dark:border-gray-700 pt-6">
      {prevPage ? (
        <Link
          href={prevPage.path}
          className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
        >
          <ChevronLeftIcon className="h-5 w-5 mr-1" />
          <div>
            <div className="text-xs uppercase tracking-wide">上一篇</div>
            <div className="mt-1 text-base font-medium">{prevPage.title}</div>
          </div>
        </Link>
      ) : (
        <div /> {/* 占位 */}
      )}

      {nextPage ? (
        <Link
          href={nextPage.path}
          className="flex items-center text-right text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
        >
          <div>
            <div className="text-xs uppercase tracking-wide">下一篇</div>
            <div className="mt-1 text-base font-medium">{nextPage.title}</div>
          </div>
          <ChevronRightIcon className="h-5 w-5 ml-1" />
        </Link>
      ) : (
        <div /> {/* 占位 */}
      )}
    </div>
  );
} 
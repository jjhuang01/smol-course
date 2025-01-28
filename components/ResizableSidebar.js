import { useState, useEffect, useCallback, useRef } from 'react'

const STORAGE_KEY = 'sidebar-width'

export default function ResizableSidebar({ children, minWidth = 240, maxWidth = 400, defaultWidth = 280 }) {
  // 使用 ref 存储 DOM 元素
  const sidebarRef = useRef(null)
  const [isResizing, setIsResizing] = useState(false)
  
  // 从本地存储获取初始宽度
  const [width, setWidth] = useState(() => {
    if (typeof window === 'undefined') return defaultWidth
    return Number(localStorage.getItem(STORAGE_KEY)) || defaultWidth
  })

  // 处理鼠标按下事件
  const handleMouseDown = useCallback((e) => {
    e.preventDefault()
    const startX = e.pageX
    const startWidth = width

    function handleMouseMove(e) {
      const diff = e.pageX - startX
      const newWidth = Math.max(minWidth, Math.min(maxWidth, startWidth + diff))
      
      // 使用 requestAnimationFrame 优化性能
      requestAnimationFrame(() => {
        setWidth(newWidth)
        if (sidebarRef.current) {
          sidebarRef.current.style.width = `${newWidth}px`
        }
      })
    }

    function handleMouseUp() {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      setIsResizing(false)
      
      // 保存到本地存储
      localStorage.setItem(STORAGE_KEY, width.toString())
    }

    setIsResizing(true)
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [width, minWidth, maxWidth])

  // 双击重置宽度
  const handleDoubleClick = useCallback(() => {
    setWidth(defaultWidth)
    localStorage.setItem(STORAGE_KEY, defaultWidth.toString())
    if (sidebarRef.current) {
      sidebarRef.current.style.width = `${defaultWidth}px`
    }
  }, [defaultWidth])

  // 组件卸载时保存宽度
  useEffect(() => {
    return () => {
      localStorage.setItem(STORAGE_KEY, width.toString())
    }
  }, [width])

  return (
    <div
      ref={sidebarRef}
      className={`relative flex-shrink-0 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 transition-colors ${
        isResizing ? 'select-none' : ''
      }`}
      style={{ 
        width: `${width}px`,
        willChange: isResizing ? 'width' : 'auto'
      }}
    >
      <div className="h-full flex flex-col">
        <div className="flex-shrink-0">
          {/* 固定在顶部的内容 */}
          {children[0]}
        </div>
        <div className="flex-1 overflow-y-auto sidebar-content">
          {/* 可滚动的内容 */}
          {children[1]}
        </div>
      </div>
      
      <div
        className={`absolute top-0 right-0 w-1 h-full cursor-col-resize group ${
          isResizing ? 'bg-primary-500/50' : 'hover:bg-primary-500/30'
        }`}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
      >
        {/* 拖动把手 */}
        <div className={`absolute inset-y-0 -right-1 w-3 group-hover:bg-primary-500/10 transition-colors ${
          isResizing ? 'bg-primary-500/20' : ''
        }`} />
      </div>
      
      {/* 拖动时的全局遮罩 */}
      {isResizing && (
        <div className="fixed inset-0 z-50 cursor-col-resize" />
      )}
    </div>
  )
} 
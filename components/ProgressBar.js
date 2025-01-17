import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';

export default function ProgressBar() {
  const [progress, setProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const router = useRouter();

  useEffect(() => {
    let timer;

    const handleStart = () => {
      setIsVisible(true);
      setProgress(0);
      timer = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(timer);
            return 90;
          }
          return prev + 10;
        });
      }, 100);
    };

    const handleComplete = () => {
      clearInterval(timer);
      setProgress(100);
      setTimeout(() => {
        setIsVisible(false);
        setProgress(0);
      }, 300);
    };

    router.events.on('routeChangeStart', handleStart);
    router.events.on('routeChangeComplete', handleComplete);
    router.events.on('routeChangeError', handleComplete);

    return () => {
      clearInterval(timer);
      router.events.off('routeChangeStart', handleStart);
      router.events.off('routeChangeComplete', handleComplete);
      router.events.off('routeChangeError', handleComplete);
    };
  }, [router]);

  if (!isVisible) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-50">
      <div
        className="h-1 bg-primary-500"
        style={{
          width: `${progress}%`,
          transition: 'width 0.2s ease-in-out',
        }}
      />
    </div>
  );
} 
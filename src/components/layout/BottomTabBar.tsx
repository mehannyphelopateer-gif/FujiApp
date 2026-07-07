import type { ReactElement } from "react";

interface Tab {
  path: string;
  label: string;
  icon: ReactElement;
}

const TABS: Tab[] = [
  {
    path: "/",
    label: "Preview",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" className="h-5 w-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2 4 4 8-8 4 4" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 19h18" />
      </svg>
    ),
  },
  {
    path: "/recipes",
    label: "Recipes",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" className="h-5 w-5">
        <rect x="3.5" y="3.5" width="7" height="7" rx="1" />
        <rect x="13.5" y="3.5" width="7" height="7" rx="1" />
        <rect x="3.5" y="13.5" width="7" height="7" rx="1" />
        <rect x="13.5" y="13.5" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    path: "/compare",
    label: "Compare",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" className="h-5 w-5">
        <rect x="3" y="4" width="8" height="16" rx="1" />
        <rect x="13" y="4" width="8" height="16" rx="1" />
      </svg>
    ),
  },
  {
    path: "/batch",
    label: "Batch",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" className="h-5 w-5">
        <rect x="3.5" y="3.5" width="13" height="13" rx="1.5" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 20.5h13v-13" />
      </svg>
    ),
  },
  {
    path: "/ai",
    label: "AI",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" className="h-5 w-5">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 3l1.8 4.6L18 9.3l-4.2 1.7L12 15.5l-1.8-4.5L6 9.3l4.2-1.7L12 3Z"
        />
        <path strokeLinecap="round" strokeLinejoin="round" d="M19 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8.8-2Z" />
      </svg>
    ),
  },
];

interface BottomTabBarProps {
  path: string;
  onNavigate: (to: string) => void;
}

export function BottomTabBar({ path, onNavigate }: BottomTabBarProps) {
  return (
    <nav
      role="tablist"
      aria-label="Primary"
      className="flex shrink-0 border-t border-ink-800 bg-ink-950 [padding-bottom:env(safe-area-inset-bottom)] [padding-left:env(safe-area-inset-left)] [padding-right:env(safe-area-inset-right)]"
    >
      {TABS.map((tab) => {
        const isActive = tab.path === path;
        return (
          <button
            key={tab.path}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onNavigate(tab.path)}
            className="relative flex flex-1 flex-col items-center gap-1 py-2.5 text-[10px] font-bold uppercase tracking-wide transition-colors"
          >
            <span className={isActive ? "text-gold-400" : "text-ink-500"}>{tab.icon}</span>
            <span className={isActive ? "text-gold-400" : "text-ink-500"}>{tab.label}</span>
            {isActive && <span className="absolute -top-px h-0.5 w-8 bg-gold-400" />}
          </button>
        );
      })}
    </nav>
  );
}

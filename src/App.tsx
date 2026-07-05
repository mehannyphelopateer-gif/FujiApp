import { AppStateProvider } from "@/context/AppStateContext";
import { AppShell } from "@/components/layout/AppShell";
import { RecipesPage } from "@/components/recipes-page/RecipesPage";
import { BottomTabBar } from "@/components/layout/BottomTabBar";
import { useRoute } from "@/hooks/useRoute";

export default function App() {
  const { path, navigate } = useRoute();

  return (
    <AppStateProvider>
      <div className="flex h-dvh w-full flex-col bg-ink-950 text-ink-50 [padding-left:env(safe-area-inset-left)] [padding-right:env(safe-area-inset-right)]">
        <div className="min-h-0 flex-1">{path === "/recipes" ? <RecipesPage /> : <AppShell />}</div>
        <BottomTabBar path={path} onNavigate={navigate} />
      </div>
    </AppStateProvider>
  );
}

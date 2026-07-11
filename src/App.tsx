import { AppStateProvider } from "@/context/AppStateContext";
import { CameraLinkProvider } from "@/context/CameraLinkContext";
import { AppShell } from "@/components/layout/AppShell";
import { RecipesPage } from "@/components/recipes-page/RecipesPage";
import { CompareView } from "@/components/compare/CompareView";
import { BatchView } from "@/components/batch/BatchView";
import { AIPage } from "@/components/ai/AIPage";
import { CameraDebugPage } from "@/components/camera/CameraDebugPage";
import { BottomTabBar } from "@/components/layout/BottomTabBar";
import { useRoute } from "@/hooks/useRoute";

export default function App() {
  const { path, navigate } = useRoute();

  return (
    <AppStateProvider>
      <CameraLinkProvider>
        <div className="flex h-dvh w-full flex-col bg-ink-950 text-ink-50 [padding-left:env(safe-area-inset-left)] [padding-right:env(safe-area-inset-right)]">
          <div className="min-h-0 flex-1">
            {path === "/recipes" ? (
              <RecipesPage />
            ) : path === "/compare" ? (
              <CompareView />
            ) : path === "/batch" ? (
              <BatchView />
            ) : path === "/ai" ? (
              <AIPage />
            ) : path === "/camera" ? (
              <CameraDebugPage />
            ) : (
              <AppShell />
            )}
          </div>
          <BottomTabBar path={path} onNavigate={navigate} />
        </div>
      </CameraLinkProvider>
    </AppStateProvider>
  );
}

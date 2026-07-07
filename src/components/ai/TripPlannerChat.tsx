import { useRef, useState, useEffect } from "react";
import { useAppState } from "@/context/AppStateContext";
import { sendTripChatMessage, type ChatMessage, type TripRecommendation } from "@/lib/ai/tripRecipeChat";

interface DisplayMessage extends ChatMessage {
  recommendations?: TripRecommendation[];
}

const GREETING: DisplayMessage = {
  role: "assistant",
  content: "Where are you headed? Tell me the destination, and roughly when you're going if you know — I'll put together 7 recipes to load onto your camera.",
};

export function TripPlannerChat() {
  const { recipes, setSelectedRecipeId } = useAppState();
  const [messages, setMessages] = useState<DisplayMessage[]>([GREETING]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  const recipeById = new Map(recipes.map((r) => [r.id, r]));

  async function handleSend() {
    const text = input.trim();
    if (!text || isLoading) return;

    const nextMessages: DisplayMessage[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setIsLoading(true);

    try {
      const result = await sendTripChatMessage(nextMessages.map(({ role, content }) => ({ role, content })));
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: result.reply, recommendations: result.recommendations },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.map((message, index) => (
          <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-lg px-3.5 py-2.5 text-sm ${
                message.role === "user" ? "bg-gold-500 text-ink-950" : "border border-ink-800 bg-ink-900 text-ink-100"
              }`}
            >
              <p className="leading-relaxed">{message.content}</p>

              {message.recommendations && message.recommendations.length > 0 && (
                <div className="mt-3 space-y-2 border-t border-ink-700/60 pt-3">
                  {message.recommendations.map((rec, i) => {
                    const recipe = recipeById.get(rec.recipeId);
                    if (!recipe) return null;
                    return (
                      <div key={rec.recipeId} className="rounded border border-ink-700 bg-ink-950/40 p-2.5">
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-bold text-ink-50">
                            {i + 1}. {recipe.name}
                          </p>
                          <button
                            type="button"
                            onClick={() => setSelectedRecipeId(recipe.id)}
                            className="shrink-0 rounded bg-gold-500/20 px-2 py-1 text-[10px] font-bold uppercase text-gold-300 transition-colors hover:bg-gold-500/30"
                          >
                            Apply
                          </button>
                        </div>
                        <p className="mt-1 text-[11px] text-ink-400">{rec.reasoning}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 rounded-lg border border-ink-800 bg-ink-900 px-3.5 py-2.5">
              <svg className="h-3.5 w-3.5 animate-spin text-gold-400" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              <span className="text-xs text-ink-500">Thinking…</span>
            </div>
          </div>
        )}

        {error && <p className="rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}
      </div>

      <div className="flex shrink-0 gap-2 border-t border-ink-800 p-3">
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void handleSend();
            }
          }}
          placeholder="e.g. Spain and Egypt in the fall…"
          disabled={isLoading}
          className="flex-1 rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-50 placeholder:text-ink-500 focus:border-gold-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={() => void handleSend()}
          disabled={isLoading || !input.trim()}
          className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 transition-colors hover:bg-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}

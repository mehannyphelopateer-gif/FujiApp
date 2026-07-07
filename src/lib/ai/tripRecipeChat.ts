export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface TripRecommendation {
  recipeId: string;
  reasoning: string;
}

export interface TripChatResult {
  reply: string;
  recommendations: TripRecommendation[];
}

export async function sendTripChatMessage(messages: ChatMessage[]): Promise<TripChatResult> {
  const response = await fetch("/api/trip-recipe-chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages }),
  });

  if (!response.ok) {
    const errorBody = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(errorBody?.error ?? `Request failed with status ${response.status}`);
  }

  return response.json();
}

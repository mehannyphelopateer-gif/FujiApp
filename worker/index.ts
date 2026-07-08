import { handleRecommendRecipe } from "./handlers/recommend-recipe";
import { handleTripRecipeChat } from "./handlers/trip-recipe-chat";

interface Env {
  ANTHROPIC_API_KEY?: string;
  ASSETS: Fetcher;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/api/recommend-recipe") {
      return handleRecommendRecipe(request, env);
    }
    if (request.method === "POST" && url.pathname === "/api/trip-recipe-chat") {
      return handleTripRecipeChat(request, env);
    }

    return env.ASSETS.fetch(request);
  },
};

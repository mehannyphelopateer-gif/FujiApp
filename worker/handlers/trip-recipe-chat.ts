import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";
import recipesData from "../../src/data/recipes.json";

interface CatalogEntry {
  id: string;
  name: string;
  baseFilmSimulation: string;
  description?: string;
}

const catalog: CatalogEntry[] = (recipesData as CatalogEntry[]).map((r) => ({
  id: r.id,
  name: r.name,
  baseFilmSimulation: r.baseFilmSimulation,
  description: r.description,
}));
const catalogIds = new Set(catalog.map((r) => r.id));

const catalogText = catalog
  .map(
    (r) =>
      `- id: "${r.id}" | name: "${r.name}" | base film simulation: ${r.baseFilmSimulation}${
        r.description ? ` | ${r.description}` : ""
      }`,
  )
  .join("\n");

const SYSTEM_PROMPT =
  "You are a Fujifilm film simulation travel consultant. A photographer is planning a trip and " +
  "wants help picking which recipes to load onto their camera (Fuji cameras store a handful of " +
  "custom presets, so they need a focused shortlist, not the whole catalog). Have a short, " +
  "natural conversation to understand their trip — destination, season/time of year, and general " +
  "vibe (e.g. beach town vs. old city streets vs. desert) are the most useful things to know. Ask " +
  "at most one or two clarifying questions if the trip is vague (e.g. \"Egypt\" alone doesn't say " +
  "much about season or subject matter), but don't stall indefinitely — once you have a reasonable " +
  "sense of the trip, recommend exactly 7 recipes from the catalog below by id, ordered best-first, " +
  "each with a short reason tied to that destination's light/color/mood. Only recommend ids that " +
  `appear in this catalog — never invent one.\n\nCatalog:\n${catalogText}`;

const ChatResponseSchema = z.object({
  reply: z.string().describe("Conversational reply shown to the user — a clarifying question, or a short intro to the recommendations."),
  recommendations: z
    .array(
      z.object({
        recipeId: z.string().describe("Must exactly match an id from the provided catalog — never invent one."),
        reasoning: z.string().describe("One sentence on why this recipe suits this specific trip."),
      }),
    )
    .describe("Exactly 7 recipes once you have enough info about the trip — empty array while still clarifying."),
});

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const MAX_MESSAGES = 20;
const MAX_MESSAGE_LENGTH = 2000;

export interface Env {
  ANTHROPIC_API_KEY?: string;
}

export async function handleTripRecipeChat(request: Request, env: Env): Promise<Response> {
  const apiKey = env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return jsonResponse(500, { error: "Server is not configured with an Anthropic API key." });
  }

  let body: { messages?: ChatMessage[] };
  try {
    body = (await request.json()) as { messages?: ChatMessage[] };
  } catch {
    return jsonResponse(400, { error: "Invalid JSON body." });
  }

  const { messages } = body;
  if (!Array.isArray(messages) || messages.length === 0) {
    return jsonResponse(400, { error: "messages is required and must be non-empty." });
  }
  if (messages.length > MAX_MESSAGES) {
    return jsonResponse(413, { error: "Conversation is too long." });
  }
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant") {
      return jsonResponse(400, { error: "Each message must have role 'user' or 'assistant'." });
    }
    if (typeof m.content !== "string" || m.content.length === 0 || m.content.length > MAX_MESSAGE_LENGTH) {
      return jsonResponse(400, { error: "Each message must have non-empty content within the length limit." });
    }
  }

  const client = new Anthropic({ apiKey });

  try {
    const response = await client.messages.parse({
      model: "claude-opus-4-8",
      max_tokens: 2048,
      system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      output_config: { format: zodOutputFormat(ChatResponseSchema) },
    });

    console.log(
      `trip-recipe-chat usage: input=${response.usage.input_tokens} cache_write=${response.usage.cache_creation_input_tokens ?? 0} cache_read=${response.usage.cache_read_input_tokens ?? 0} output=${response.usage.output_tokens}`,
    );

    if (!response.parsed_output) {
      return jsonResponse(502, { error: "Claude's response didn't match the expected format." });
    }

    const recommendations = response.parsed_output.recommendations.filter((r) => catalogIds.has(r.recipeId));

    return jsonResponse(200, {
      reply: response.parsed_output.reply,
      recommendations,
    });
  } catch (err) {
    console.error("trip-recipe-chat error", err);
    return jsonResponse(502, { error: "Failed to get a response from Claude." });
  }
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

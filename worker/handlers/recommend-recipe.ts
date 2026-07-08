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

// A slimmed-down view of the recipe catalog — Claude only needs enough to
// reason about *which look* fits the scene, not the exact numeric shader
// parameters (WB shift, tone curve, etc).
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

// This never changes between requests (same recipe catalog every time), so
// it lives in `system` with a cache breakpoint rather than being re-sent as
// part of the per-request user message — repeat requests within the
// 5-minute TTL pay ~0.1x for these tokens instead of full price. Only the
// photo itself (genuinely different every request) stays uncached.
const SYSTEM_PROMPT =
  "You are a Fujifilm film simulation expert helping a photographer pick the best \"recipe\" " +
  "(a Fuji X Weekly-style film simulation preset) for the photo shown. Look at the lighting, " +
  "color palette, subject, and mood of the scene, then recommend the 1-3 best-matching recipes " +
  "from the catalog below by id. Only recommend ids that appear in this catalog — never invent " +
  `one.\n\nCatalog:\n${catalogText}`;

const RecommendationSchema = z.object({
  sceneDescription: z.string().describe("One or two sentences on the scene's lighting, color palette, and mood."),
  recommendations: z
    .array(
      z.object({
        recipeId: z.string().describe("Must exactly match an id from the provided catalog — never invent one."),
        reasoning: z.string().describe("One or two sentences on why this recipe suits this specific scene."),
      }),
    )
    .describe("1-3 recipes from the catalog, best match first."),
});

const MAX_BASE64_LENGTH = 15_000_000; // ~11MB raw — generous given the client downscales before sending.

export interface Env {
  ANTHROPIC_API_KEY?: string;
}

export async function handleRecommendRecipe(request: Request, env: Env): Promise<Response> {
  const apiKey = env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return jsonResponse(500, { error: "Server is not configured with an Anthropic API key." });
  }

  let body: { imageBase64?: string; mimeType?: string };
  try {
    body = (await request.json()) as { imageBase64?: string; mimeType?: string };
  } catch {
    return jsonResponse(400, { error: "Invalid JSON body." });
  }

  const { imageBase64, mimeType } = body;
  if (!imageBase64 || !mimeType) {
    return jsonResponse(400, { error: "imageBase64 and mimeType are required." });
  }
  if (mimeType !== "image/jpeg" && mimeType !== "image/png") {
    return jsonResponse(400, { error: "Only image/jpeg and image/png are supported." });
  }
  if (imageBase64.length > MAX_BASE64_LENGTH) {
    return jsonResponse(413, { error: "Image is too large." });
  }

  const client = new Anthropic({ apiKey });

  try {
    const response = await client.messages.parse({
      model: "claude-opus-4-8",
      max_tokens: 2048,
      system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: { type: "base64", media_type: mimeType, data: imageBase64 },
            },
          ],
        },
      ],
      output_config: { format: zodOutputFormat(RecommendationSchema) },
    });

    console.log(
      `recommend-recipe usage: input=${response.usage.input_tokens} cache_write=${response.usage.cache_creation_input_tokens ?? 0} cache_read=${response.usage.cache_read_input_tokens ?? 0} output=${response.usage.output_tokens}`,
    );

    if (!response.parsed_output) {
      return jsonResponse(502, { error: "Claude's response didn't match the expected format." });
    }

    // Defense in depth: the prompt says "never invent one", but strip any
    // hallucinated id before it reaches the client rather than trusting the
    // model's compliance.
    const recommendations = response.parsed_output.recommendations.filter((r) => catalogIds.has(r.recipeId));

    return jsonResponse(200, {
      sceneDescription: response.parsed_output.sceneDescription,
      recommendations,
    });
  } catch (err) {
    console.error("recommend-recipe error", err);
    return jsonResponse(502, { error: "Failed to get a recommendation from Claude." });
  }
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

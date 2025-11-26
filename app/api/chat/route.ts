export const runtime = "nodejs";

import { NextRequest } from "next/server"; // if using Next 13+ edge, else use Request
import sharp from "sharp";
import { Buffer } from "buffer"; // Node Buffer
import OpenAI from "openai";
import { streamText, UIMessage, convertToModelMessages, stepCountIs, createUIMessageStream, createUIMessageStreamResponse } from "ai";
import { MODEL } from "@/config";
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ----- IMAGE UPLOAD FLOW -----
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;
    if (!file) return new Response(JSON.stringify({ response: "No image provided" }), { status: 400 });

    // create Node Buffer safely
    const ab = await file.arrayBuffer();
    const u8 = new Uint8Array(ab);
    const buffer = Buffer.from(u8);

    // Optional: small server-side enhancement (be conservative)
    let enhancedBuffer = buffer;
    try {
      enhancedBuffer = await sharp(buffer)
        .rotate() // auto rotate
        .resize({ width: 1600, withoutEnlargement: true })
        .grayscale()
        .sharpen()
        .normalize()
        .jpeg({ quality: 82 })
        .toBuffer();
    } catch (err) {
      enhancedBuffer = buffer; // fallback
    }

    // convert to data-uri for the model
    const dataUrl = `data:${file.type};base64,${enhancedBuffer.toString("base64")}`;

    // ---------- OCR step: ask model to extract ONLY ingredient list ----------
    const extractPrompt = `
You are an OCR assistant: EXTRACT ONLY the ingredient list from the supplied product label image.
Return plain, comma-separated ingredients or newline-separated ingredients — nothing else.
If you cannot find an "Ingredients" section, extract any text that looks like a list of ingredients.

<image>${dataUrl}</image>
`;

    const extractRes = await openai.responses.create({
      model: "gpt-4.1-mini",
      input: extractPrompt,
    });

    const extracted = (extractRes.output_text || "").trim();

    // ---------- Analysis step: classify each ingredient using Indian rules ----------
    // We use a strong system + analyzer prompt (see prompts/cosmetics-system.ts in repo)
    const analysisPrompt = `
You are an Indian cosmetics ingredient safety analyst.

Task:
1) Parse the following ingredient list and produce a structured classification for each ingredient.
2) For each ingredient, output:
   - name (INCI if possible)
   - classification: one of [SAFE, POTENTIALLY_IRRITATING, RESTRICTED/BANNED, PREGNANCY_UNSAFE, KID_UNSAFE, COMEDOGENIC]
   - short reason (1 sentence)
   - suggested caution (if any)
3) At the end, provide an overall safety score (0-10) and a short recommendation for Indian users (skin type/humidity/usage tips).

Indian specifics to follow:
- Flag mercury compounds and lead compounds as BANNED.
- Flag hydroquinone as RESTRICTED/BANNED (unless prescribed).
- Parabens: SAFE in low concentrations but mention caution for sensitive users.
- Phenoxyethanol: SAFE under 1%.
- Formulate guidance on fragrances (common irritant in Indian climate).
- For comedogenicity mention common listed oils (Cocos Nucifera / Coconut, Butyrospermum Parkii / Shea, etc.)
- Mention pregnancy rules: retinoids, salicylic acid (high concentration), hydroquinone -> PREGNANCY_UNSAFE.

Ingredient list:
${extracted}

Return only JSON with fields:
{
  "extracted": "...",
  "items": [
    { "name": "...", "classification": "...", "reason": "...", "caution": "..." }
  ],
  "score": 7.1,
  "recommendation": "..."
}
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4.1-mini",
      input: analysisPrompt,
      // temperature low for repeatability:
      temperature: 0.0,
      max_output_tokens: 1000,
    });

    const jsonText = analysisRes.output_text?.trim() || "";
    // Model might produce additional text — attempt to parse JSON out of it.
    // If the model returned plain JSON, parse. Otherwise, wrap fallback.
    let parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch (e) {
      // fallback: wrap results into a simple object
      parsed = {
        extracted,
        items: [{ name: extracted, classification: "UNKNOWN", reason: "Could not parse structured output", caution: "" }],
        score: 5.0,
        recommendation: "Could not parse detailed analysis."
      };
    }

    return new Response(JSON.stringify({ response: JSON.stringify(parsed, null, 2), raw: parsed }), { status: 200 });
  }

  // ----- NORMAL TEXT CHAT FLOW (existing) -----
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latestUserMessage = messages?.filter((m) => m.role === "user").pop();
  if (latestUserMessage) {
    const textParts = latestUserMessage.parts?.filter(p => p.type === "text").map(p => ("text" in p ? p.text : "")).join("") || "";
    if (textParts) {
      const moderationResult = await isContentFlagged(textParts);
      if (moderationResult.flagged) {
        const stream = createUIMessageStream({
          execute({ writer }) {
            const textId = "moderation-denial-text";
            writer.write({ type: "start" });
            writer.write({ type: "text-start", id: textId });
            writer.write({ type: "text-delta", id: textId, delta: moderationResult.denialMessage || "Your message violates our guidelines." });
            writer.write({ type: "text-end", id: textId });
            writer.write({ type: "finish" });
          }
        });
        return createUIMessageStreamResponse({ stream });
      }
    }
  }

  const result = streamText({
    model: MODEL,
    system: SYSTEM_PROMPT,
    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    stopWhen: stepCountIs(10),
  });

  return result.toUIMessageStreamResponse({ sendReasoning: true });
}

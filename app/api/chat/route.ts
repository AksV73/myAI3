// /app/api/chat/route.ts
export const runtime = "nodejs";
export const preferredRegion = "bom1";

import OpenAI from "openai";
import sharp from "sharp";
import { Buffer } from "buffer";

import {
  streamText,
  UIMessage,
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs
} from "ai";

import { MODEL } from "@/config";
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

export const maxDuration = 30;

/**
 * Try to extract a JSON substring from arbitrary model text.
 * - Look for fenced code blocks first (```json ... ``` or ``` ... ```)
 * - Otherwise find the first top-level '{ ... }' block.
 */
function extractJSON(text: string | undefined): any | null {
  if (!text) return null;
  // 1) Try fenced block ```json ... ``` or ```
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  let candidate = fenceMatch ? fenceMatch[1] : text;

  // 2) Try to find the first JSON object {...}
  const objMatch = candidate.match(/\{[\s\S]*\}$/m) || candidate.match(/\{[\s\S]*?\}/m);
  if (!objMatch) return null;

  const jsonStr = objMatch[0];

  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    // Attempt to repair common issues: replace smart quotes and trailing commas
    const repaired = jsonStr
      .replace(/[“”]/g, '"')
      .replace(/,(\s*[}\]])/g, "$1"); // remove trailing commas
    try {
      return JSON.parse(repaired);
    } catch (e2) {
      return null;
    }
  }
}

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ---------- IMAGE MODE: OCR + Analysis ----------
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return Response.json({ ok: false, error: "No image uploaded." }, { status: 400 });
    }

    // Convert File => Node Buffer safely
    const ab = await file.arrayBuffer();
    const u8 = new Uint8Array(ab);
    const nodeBuffer = Buffer.from(u8);

    // Use sharp safely in Vercel + cast to any to avoid typing error
    let processedBuffer = nodeBuffer;
    try {
      processedBuffer = await (sharp as any)(nodeBuffer as any).rotate().toBuffer();
    } catch (err) {
      // fallback to original buffer
      processedBuffer = nodeBuffer;
    }

    const dataUrl = `data:${file.type};base64,${processedBuffer.toString("base64")}`;

    // ---------- OCR extraction ----------
    const ocrPrompt = `
Extract ONLY the cosmetic ingredient list from this image.

Rules:
- Return only the text that follows "Ingredients:" (or equivalent heading).
- Do NOT add explanations.
- If you cannot locate or read an ingredient list, reply: UNREADABLE
<image>${dataUrl}</image>
`;

    const ocrRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      input: ocrPrompt
    });

    const ocrText = (ocrRes.output_text || "").trim();

    if (!ocrText || ocrText.toUpperCase().includes("UNREADABLE")) {
      return Response.json({
        ok: true,
        extracted: null,
        message: "OCR could not extract a readable ingredient list from the image."
      });
    }

    // Sanitize basic result
    const extractedIngredients = ocrText;

    // ---------- Analysis prompt (request JSON) ----------
    const analysisPrompt = `
You are an Indian cosmetic ingredient safety evaluator.

Input: the ingredient list below (one string).
Produce STRICT JSON ONLY with this format:

{
  "ingredients": [
    { "name": "...", "classification": "...", "reason": "..." }
  ],
  "score": 0,
  "summary": "..."
}

Classification values: SAFE, IRRITANT, RESTRICTED/BANNED, PREGNANCY_UNSAFE, KID_UNSAFE, COMEDOGENIC

Indian specifics:
- Flag mercury/lead compounds as RESTRICTED/BANNED.
- Hydroquinone: RESTRICTED/BANNED for OTC.
- Parabens: mention "use with caution" rather than panic.
- Phenoxyethanol: OK under 1% (mention if uncertain).
- Fragrances: often IRRITANT.
Return only JSON (no text outside).
Ingredient list:
"""${extractedIngredients}"""
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      max_output_tokens: 1500,
      input: analysisPrompt
    });

    const analysisText = analysisRes.output_text || "";

    // Try to extract JSON (robust)
    let parsed = extractJSON(analysisText);

    if (!parsed) {
      // If model didn't give good JSON, attempt a fallback: ask again with a stricter instruction but include previous text
      const repairPrompt = `
The previous attempt returned this text:
"""${analysisText}"""

It didn't parse as JSON. Please REPLY WITH STRICT JSON in the exact format:

{
  "ingredients": [ { "name":"", "classification":"", "reason":"" } ],
  "score": 0,
  "summary": ""
}
`;
      const repairRes = await openai.responses.create({
        model: "gpt-4o-mini",
        temperature: 0,
        max_output_tokens: 1200,
        input: repairPrompt
      });
      parsed = extractJSON(repairRes.output_text || "") || null;
    }

    // Final fallback if still not parsable
    if (!parsed) {
      return Response.json({
        ok: true,
        extracted: extractedIngredients,
        parsed: null,
        message: "Could not parse analyzer JSON. See raw analyzer output.",
        raw: analysisText
      });
    }

    // Success
    return Response.json({
      ok: true,
      extracted: extractedIngredients,
      analysis: parsed
    });
  }

  // ---------- TEXT CHAT MODE ----------
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter((m) => m.role === "user").pop();
  if (latest) {
    const textParts =
      latest.parts
        ?.filter((p) => p.type === "text")
        .map((p) => (p as any).text)
        .join("") || "";

    const moderation = await isContentFlagged(textParts);
    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id: "blocked" });
          writer.write({
            type: "text-delta",
            id: "blocked",
            delta: moderation.denialMessage || "Message blocked"
          });
          writer.write({ type: "text-end", id: "blocked" });
          writer.write({ type: "finish" });
        }
      });
      return createUIMessageStreamResponse({ stream });
    }
  }

  const result = streamText({
    model: MODEL,
    system: SYSTEM_PROMPT,
    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    stopWhen: stepCountIs(10)
  });

  return result.toUIMessageStreamResponse({
    sendReasoning: true
  });
}

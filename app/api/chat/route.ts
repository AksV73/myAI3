// =====================================================
// FORCE NODE RUNTIME (sharp requires Node.js env)
// =====================================================
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

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // =====================================================
  // 📸 IMAGE MODE — OCR + COSMETIC ANALYSIS
  // =====================================================
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return Response.json(
        { response: "No image uploaded." },
        { status: 400 }
      );
    }

    // -----------------------------------------
    // FILE → SAFE NODE BUFFER
    // -----------------------------------------
    const ab = await file.arrayBuffer();
    const buffer = Buffer.from(new Uint8Array(ab));

    // 🔥 FIX: force EXACT Node.js buffer to avoid Vercel typing errors
    const nodeBuffer = Buffer.from(buffer);

    // -----------------------------------------
    // LIGHT ENHANCEMENT (rotate only)
    // -----------------------------------------
    let enhanced = nodeBuffer;
    try {
      enhanced = await sharp(nodeBuffer).rotate().toBuffer();
    } catch {
      enhanced = nodeBuffer;
    }

    const dataUrl = `data:${file.type};base64,${enhanced.toString("base64")}`;

    // -----------------------------------------
    // OCR STEP
    // -----------------------------------------
    const ocrPrompt = `
Extract ONLY the cosmetic ingredient list from the image.

RULES:
- Only extract text after "Ingredients:" or similar.
- Do NOT guess ingredients.
- No extra words.
- If unreadable, return "UNREADABLE".

<image>${dataUrl}</image>
`;

    const ocrRes = await openai.responses.create({
      model: "gpt-4o-mini",
      input: ocrPrompt,
      temperature: 0
    });

    const extracted = ocrRes.output_text?.trim() || "UNREADABLE";

    // -----------------------------------------
    // SAFETY ANALYSIS STEP
    // -----------------------------------------
    const analysisPrompt = `
You are an Indian cosmetics ingredient safety evaluator.

Analyze the following ingredient list:

"${extracted}"

Return STRICT JSON ONLY:

{
  "ingredients": [
    { "name": "", "classification": "", "reason": "" }
  ],
  "score": 0,
  "summary": ""
}

Classification rules:
- SAFE
- IRRITANT (fragrance, sulfates, essential oils)
- RESTRICTED/BANNED (mercury, lead, hydroquinone OTC)
- PREGNANCY_UNSAFE (retinoids, strong BHA/AHA)
- KID_UNSAFE (MIT/CMIT, heavy fragrances)
- COMEDOGENIC (coconut oil, cocoa butter, shea butter)
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4o-mini",
      input: analysisPrompt,
      temperature: 0,
      max_output_tokens: 1500
    });

    let parsed;
    try {
      parsed = JSON.parse(analysisRes.output_text || "{}");
    } catch {
      parsed = {
        ingredients: [],
        score: 5,
        summary: "Could not parse output JSON.",
        raw: analysisRes.output_text
      };
    }

    return Response.json({
      response: JSON.stringify(parsed, null, 2)
    });
  }

  // =====================================================
  // 💬 NORMAL TEXT CHAT MODE
  // =====================================================

  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const text =
      latest.parts
        ?.filter((p) => p.type === "text")
        .map((p) => ("text" in p ? p.text : ""))
        .join("") || "";

    const moderation = await isContentFlagged(text);

    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id: "blocked" });
          writer.write({
            type: "text-delta",
            id: "blocked",
            delta: moderation.denialMessage || "Message blocked."
          });
          writer.write({ type: "text-end", id: "blocked" });
          writer.write({ type: "finish" });
        }
      });
      return createUIMessageStreamResponse({ stream });
    }
  }

  // — Streaming chat continues normally —
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

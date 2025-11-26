export const runtime = "nodejs";

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

  // ---------------------------------------------------------------------
  // 📸 IMAGE MODE — Cosmetic Ingredient OCR + Safety Analysis
  // ---------------------------------------------------------------------
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;
    if (!file)
      return Response.json({ response: "No image uploaded." }, { status: 400 });

    // Convert to Node Buffer
    const u8 = new Uint8Array(await file.arrayBuffer());
    const buffer = Buffer.from(u8);

    // Light rotate only (NO resizing, NO sharpening)
    let enhanced = buffer;
    try {
      enhanced = await sharp(buffer).rotate().toBuffer();
    } catch (_) {
      enhanced = buffer;
    }

    const dataUrl = `data:${file.type};base64,${enhanced.toString("base64")}`;

    // --------------------- OCR STEP ---------------------
    const ocrPrompt = `
Extract ONLY the cosmetic ingredient list from this image.
Very strict rules:
- Return ONLY the text after "Ingredients:" or equivalent.
- Do NOT guess ingredients.
- If unreadable, say "UNREADABLE".

<image>${dataUrl}</image>
`;

    const ocrRes = await openai.responses.create({
      model: "gpt-4o-mini",      // (FIXED — Strong OCR)
      temperature: 0,
      input: ocrPrompt
    });

    const extracted = ocrRes.output_text?.trim() || "UNREADABLE";

    // --------------------- ANALYSIS STEP ---------------------
    const analysisPrompt = `
You are an Indian cosmetic ingredient safety specialist.
Analyze the ingredient list below:

"${extracted}"

Return STRICT JSON ONLY:

{
  "ingredients": [
    { "name": "...", "classification": "...", "reason": "..." }
  ],
  "score": number,
  "summary": "..."
}

Rules:
- SAFE: gentle surfactants, humectants, mild preservatives.
- IRRITANT: fragrances, sulfates, essential oils.
- RESTRICTED/BANNED (India): mercury compounds, lead compounds, hydroquinone (OTC), strong steroids.
- PREGNANCY UNSAFE: retinoids, salicylic acid >2%, hydroquinone.
- KID UNSAFE: strong fragrances, sulfates, MIT/CMIT.
- COMEDOGENIC: coconut oil, shea butter, cocoa butter.
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      max_output_tokens: 1000,
      input: analysisPrompt
    });

    let parsed;
    try {
      parsed = JSON.parse(analysisRes.output_text || "{}");
    } catch {
      parsed = {
        ingredients: [],
        score: 5,
        summary: "Could not parse structured JSON.",
        raw: analysisRes.output_text
      };
    }

    return Response.json({ response: JSON.stringify(parsed, null, 2) });
  }

  // ---------------------------------------------------------------------
  // 💬 TEXT CHAT MODE (unchanged)
  // ---------------------------------------------------------------------
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();
  if (latest) {
    const textParts =
      latest.parts
        ?.filter((p) => p.type === "text")
        .map((p) => ("text" in p ? p.text : ""))
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
            delta: moderation.denialMessage || "Message blocked."
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

  return result.toUIMessageStreamResponse({ sendReasoning: true });
}

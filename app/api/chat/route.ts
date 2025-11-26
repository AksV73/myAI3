// ======================================================
// BACKEND — Cosmetic OCR + Safety Analysis
// ======================================================

export const runtime = "nodejs";
export const preferredRegion = "bom1";

import OpenAI from "openai";
import sharp from "sharp";
import { Readable } from "stream";

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

// Convert Buffer → Stream (Vercel TS-safe)
function bufferToStream(buffer: Buffer) {
  const stream = new Readable();
  stream.push(buffer);
  stream.push(null);
  return stream;
}

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ======================================================
  // 📸 IMAGE MODE
  // ======================================================
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file)
      return Response.json({ response: "No image uploaded." }, { status: 400 });

    // Convert File → Buffer
    const arr = new Uint8Array(await file.arrayBuffer());
    const buffer = Buffer.from(arr);

    // ⭐ STREAM INTO SHARP (Fixes ALL Vercel Type Errors)
    let processed = buffer;
    try {
      const stream = bufferToStream(buffer);
      processed = await sharp(stream).rotate().toBuffer();
    } catch {
      processed = buffer;
    }

    const dataUrl = `data:${file.type};base64,${processed.toString("base64")}`;

    // ---------------- OCR ----------------
    const ocrPrompt = `
Extract ONLY the cosmetic ingredients from this image:

Rules:
• Only text after "Ingredients:"
• No extras, no guesses
• If unreadable: "UNREADABLE"

<image>${dataUrl}</image>
`;

    const ocrRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      input: ocrPrompt
    });

    const extracted = ocrRes.output_text?.trim() || "UNREADABLE";

    // ---------------- ANALYSIS ----------------
    const analysisPrompt = `
You are an Indian cosmetic ingredient safety specialist.

Analyze:

"${extracted}"

Return STRICT JSON ONLY:
{
  "ingredients": [
    { "name": "", "classification": "", "reason": "" }
  ],
  "score": 0,
  "summary": ""
}
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      max_output_tokens: 1200,
      input: analysisPrompt
    });

    let parsed;
    try {
      parsed = JSON.parse(analysisRes.output_text || "{}");
    } catch {
      parsed = {
        ingredients: [],
        score: 5,
        summary: "Could not parse JSON.",
        raw: analysisRes.output_text
      };
    }

    return Response.json({
      response: JSON.stringify(parsed, null, 2)
    });
  }

  // ======================================================
  // 💬 TEXT MODE
  // ======================================================

  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const text =
      latest.parts
        ?.filter((p) => p.type === "text")
        .map((p) => (p as any).text)
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
            delta:
              moderation.denialMessage || "Message blocked."
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

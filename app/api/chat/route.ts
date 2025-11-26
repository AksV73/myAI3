export const runtime = "nodejs";

import sharp from "sharp";
import OpenAI from "openai";
import { Buffer } from "buffer";

import {
  streamText,
  UIMessage,
  convertToModelMessages,
  stepCountIs,
  createUIMessageStream,
  createUIMessageStreamResponse,
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

  // -----------------------------------------------------------
  // IMAGE UPLOAD FLOW
  // -----------------------------------------------------------
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;
    if (!file)
      return new Response(
        JSON.stringify({ response: "No image provided" }),
        { status: 400 }
      );

    // ⭐ FIXED: Always produce a strict Node Buffer
    const arrayBuf = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuf) as Buffer;

    // ⭐ FIXED: Sharp requires explicit cast
    let enhancedBuffer: Buffer = buffer;
    try {
      enhancedBuffer = await sharp(buffer as unknown as Buffer)
        .rotate()
        .resize({ width: 1600, withoutEnlargement: true })
        .grayscale()
        .sharpen()
        .normalize()
        .jpeg({ quality: 82 })
        .toBuffer();
    } catch (err) {
      enhancedBuffer = buffer;
    }

    const dataUrl = `data:${file.type};base64,${enhancedBuffer.toString(
      "base64"
    )}`;

    // --- OCR ---
    const extractPrompt = `
Extract ONLY the ingredient list. No extra text.

<image>${dataUrl}</image>
`;

    const extractRes = await openai.responses.create({
      model: "gpt-4.1-mini",
      input: extractPrompt,
    });

    const extracted = (extractRes.output_text || "").trim();

    // --- ANALYSIS ---
    const analysisPrompt = `
Analyze these cosmetic ingredients using Indian BIS & cosmetic safety guidelines.

Ingredients:
${extracted}

Return structured JSON.
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4.1-mini",
      input: analysisPrompt,
      temperature: 0,
    });

    let parsed;
    try {
      parsed = JSON.parse(analysisRes.output_text || "");
    } catch {
      parsed = {
        extracted,
        error: "Could not parse structured response",
      };
    }

    return new Response(
      JSON.stringify({
        response: JSON.stringify(parsed, null, 2),
        raw: parsed,
      }),
      { status: 200 }
    );
  }

  // -----------------------------------------------------------
  // NORMAL CHAT FLOW
  // -----------------------------------------------------------
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const textParts = latest.parts
      ?.filter((p) => p.type === "text")
      .map((p) => ("text" in p ? p.text : ""))
      .join(" ");

    const moderation = await isContentFlagged(textParts);
    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id: "block" });
          writer.write({
            type: "text-delta",
            id: "block",
            delta: moderation.denialMessage || "Message blocked",
          });
          writer.write({ type: "text-end", id: "block" });
          writer.write({ type: "finish" });
        },
      });

      return createUIMessageStreamResponse({ stream });
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

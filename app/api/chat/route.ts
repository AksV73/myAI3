export const runtime = "nodejs";

import {
  streamText,
  UIMessage,
  convertToModelMessages,
  stepCountIs,
  createUIMessageStream,
  createUIMessageStreamResponse
} from "ai";

import { MODEL } from "@/config";
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";

import sharp from "sharp";

export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ---------------------------------------------------------
  // 📸 IMAGE UPLOAD MODE
  // ---------------------------------------------------------
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const formData = await req.formData();
    const file = formData.get("image") as File;
    if (!file) return Response.json({ response: "No image uploaded." });

    // --- TRUE BUFFER CREATION (NO TYPING ERRORS) ---
    const ab = await file.arrayBuffer();
    const buffer: Buffer = Buffer.from(new Uint8Array(ab));

    // --- IMAGE ENHANCEMENT WITH TYPING FIX ---
    let enhanced: Buffer = buffer;
    try {
      enhanced = await (sharp as any)(buffer)
        .rotate()
        .sharpen(0.4)
        .normalize()
        .toBuffer();
    } catch (err) {
      enhanced = buffer; // safe fallback
    }

    const dataUrl = `data:${file.type};base64,${enhanced.toString("base64")}`;

    // --- OCR ---
    const extractRes = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
Extract ONLY the ingredient list.
Return plain text only — no explanation.

<image>${dataUrl}</image>
`
    });

    const extracted = extractRes.output_text?.trim() || "Could not read ingredients.";

    // --- FSSAI ANALYSIS ---
    const analyzeRes = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You are an Indian FSSAI Additive Evaluator.

Analyze the following:
${extracted}

Classify each item as:
- SAFE
- HARMFUL
- BANNED
- KID-SENSITIVE

Give bullet points + a safety score out of 10.
`
    });

    return Response.json({
      response:
        `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
        `🔍 **FSSAI Analysis:**\n${analyzeRes.output_text}`
    });
  }

  // ---------------------------------------------------------
  // 💬 TEXT MODE
  // ---------------------------------------------------------
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();
  if (latest) {
    const text = latest.parts
      .filter((p) => p.type === "text")
      .map((p) => p.text)
      .join("");

    const moderation = await isContentFlagged(text);
    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute: ({ writer }) => {
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

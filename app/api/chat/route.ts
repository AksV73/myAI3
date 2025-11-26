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

  // ---------------------- IMAGE MODE ----------------------
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const formData = await req.formData();
    const file = formData.get("image") as File;
    if (!file) return Response.json({ response: "No image uploaded." });

    // --- FIXED BUFFER CREATION ---
    const arr = new Uint8Array(await file.arrayBuffer());
    const buffer = Buffer.from(arr);

    // --- IMAGE ENHANCEMENT ---
    let enhanced = buffer;
    try {
      enhanced = await sharp(buffer)
        .rotate()
        .sharpen(0.4)
        .normalize()
        .toBuffer();
    } catch (e) {
      enhanced = buffer; // fallback
    }

    const dataUrl = `data:${file.type};base64,${enhanced.toString("base64")}`;

    // --- OCR ---
    const extractRes = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
Extract ONLY the ingredients or any text that resembles an ingredient list.
Plain text only. No extra words.

<image>${dataUrl}</image>
`
    });

    const extracted = extractRes.output_text?.trim() || "Could not read ingredients.";

    // --- ANALYZE ---
    const analyzeRes = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You are an Indian FSSAI Additive Evaluator.

Analyze the ingredients below:
${extracted}

Classify each ingredient into:
- SAFE
- HARMFUL
- BANNED
- KID-SENSITIVE

Return structured bullet points + final score /10.
`
    });

    return Response.json({
      response:
        `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
        `🔍 **FSSAI Safety Analysis:**\n${analyzeRes.output_text}`
    });
  }

  // ---------------------- TEXT CHAT MODE ----------------------
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter(m => m.role === "user").pop();
  if (latest) {
    const textParts = latest.parts
      .filter(p => p.type === "text")
      .map(p => p.text)
      .join("");

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

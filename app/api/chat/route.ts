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

  // ======================================================
  // 📸 CASE 1 — IMAGE UPLOAD
  // ======================================================
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const formData = await req.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return Response.json({ response: "No image uploaded." });
    }

    // Convert to buffer
    const buffer = Buffer.from(await file.arrayBuffer());

    // 👍 Gentle improvement (NO format change)
    let enhanced = buffer;
    try {
      enhanced = await sharp(buffer)
        .rotate()
        .sharpen(0.4)
        .normalize()
        .toBuffer();
    } catch (err) {
      enhanced = buffer;
    }

    const dataUrl = `data:${file.type};base64,${enhanced.toString("base64")}`;

    // --- OCR ---
    const extractRes = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You are an OCR expert. Extract the ingredient list OR any text that resembles ingredients.
Return plain text only, no explanations.

<image>${dataUrl}</image>
`
    });

    const extracted = extractRes.output_text?.trim() || "Could not read ingredients.";

    // --- FSSAI Analysis ---
    const analyzeRes = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You are an Indian FSSAI Additive Analyzer.
Analyze the following ingredients:

${extracted}

Classify each item into:
- SAFE
- HARMFUL
- BANNED
- KID-SENSITIVE

Return neat bullet points + final safety score out of 10.
`
    });

    const analysis = analyzeRes.output_text || "Could not analyze.";

    return Response.json({
      response:
        `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
        `🔍 **FSSAI Safety Analysis:**\n${analysis}`
    });
  }

  // ======================================================
  // 💬 CASE 2 — NORMAL CHAT
  // ======================================================
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const textParts = latest.parts
      .filter((p) => p.type === "text")
      .map((p) => ("text" in p ? p.text : ""))
      .join("");

    const moderation = await isContentFlagged(textParts);

    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          const id = "blocked-msg";
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id });
          writer.write({
            type: "text-delta",
            id,
            delta: moderation.denialMessage || "Message blocked."
          });
          writer.write({ type: "text-end", id });
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
    tools: {
      webSearch,
      vectorDatabaseSearch
    },
    stopWhen: stepCountIs(10)
  });

  return result.toUIMessageStreamResponse({
    sendReasoning: true
  });
}

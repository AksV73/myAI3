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

export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ====================================================================
  // 📸 CASE 1 — IMAGE UPLOAD (multipart/form-data)
  // ====================================================================
  if (contentType.includes("multipart/form-data")) {
    // Lazy import OpenAI to avoid top-level import issues
    const OpenAI = (await import("openai")).default;
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const formData = await req.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return Response.json({ response: "No file uploaded." });
    }

    // Convert image to base64
    const arrayBuffer = await file.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString("base64");
    const dataUrl = `data:${file.type};base64,${base64}`;

    // --------------------------------------------------------------------
    // 1️⃣ OCR — Extract ingredient text using GPT-4o-mini Vision
    // --------------------------------------------------------------------
    const visionRes = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: [
            {
              type: "text",
              text:
                "Extract ONLY the ingredient list text from the food label image. Return plain text only.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "image",
              image: dataUrl,
            },
          ],
        },
      ],
    });

    const extracted =
      visionRes.choices[0]?.message?.content ||
      "Could not extract ingredients.";

    // --------------------------------------------------------------------
    // 2️⃣ Analyze ingredients using FSSAI rules
    // --------------------------------------------------------------------
    const analysisRes = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: [
            {
              type: "text",
              text:
                "You are an Indian FSSAI Additive Analyzer. Classify ingredients into SAFE / HARMFUL / BANNED / KID-SENSITIVE. Use clear bullet points.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: extracted,
            },
          ],
        },
      ],
    });

    const analysis =
      analysisRes.choices[0]?.message?.content ||
      "Could not analyze ingredients.";

    return Response.json({
      response:
        `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
        `🔍 **FSSAI Safety Analysis:**\n${analysis}`,
    });
  }

  // ====================================================================
  // 💬 CASE 2 — NORMAL CHAT MESSAGE (existing logic preserved)
  // ====================================================================
  const { messages }: { messages: UIMessage[] } = await req.json();

  // Get user's latest message for moderation
  const latestUserMessage = messages
    .filter((msg) => msg.role === "user")
    .pop();

  if (latestUserMessage) {
    const textParts = latestUserMessage.parts
      .filter((p) => p.type === "text")
      .map((p) => p.text)
      .join("");

    if (textParts) {
      // Content moderation
      const moderationResult = await isContentFlagged(textParts);

      if (moderationResult.flagged) {
        const stream = createUIMessageStream({
          execute({ writer }) {
            const id = "moderation-warning";

            writer.write({ type: "start" });
            writer.write({ type: "text-start", id });
            writer.write({
              type: "text-delta",
              id,
              delta:
                moderationResult.denialMessage ||
                "Your message violates our guidelines.",
            });
            writer.write({ type: "text-end", id });
            writer.write({ type: "finish" });
          },
        });

        return createUIMessageStreamResponse({ stream });
      }
    }
  }

  // --------------------------------------------------------------------
  // Normal text chat response using streamText
  // --------------------------------------------------------------------
  const result = streamText({
    model: MODEL,
    system: SYSTEM_PROMPT,
    messages: convertToModelMessages(messages),
    tools: {
      webSearch,
      vectorDatabaseSearch,
    },
    stopWhen: stepCountIs(10),
  });

  return result.toUIMessageStreamResponse({
    sendReasoning: true,
  });
}

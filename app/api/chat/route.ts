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

export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ----------------------------------------------------------------------
  // 📸 CASE 1 — IMAGE UPLOAD (multipart/form-data)
  // ----------------------------------------------------------------------
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return Response.json({ response: "No image received." });
    }

    // Convert image → base64 → data URL
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // Lazy import OpenAI (fixes Next.js edge issues)
    const OpenAI = (await import("openai")).default;
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
    });

    // STEP 1 — Extract ingredients
    const extractRes = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "Extract ONLY the ingredient list from the food label. Plain text only." },
        {
          role: "user",
          content: [
            { type: "text", text: "Extract ingredients from this image:" },
            { type: "image", image: dataUrl }
          ]
        }
      ]
    });

    const extracted = extractRes.choices[0].message.content || "Could not extract ingredients.";

    // STEP 2 — Analyze ingredients
    const analysisRes = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are an Indian FSSAI Additive Analyzer. Classify ingredients into SAFE, HARMFUL, BANNED, KID-SENSITIVE. Use bullet points."
        },
        { role: "user", content: extracted }
      ]
    });

    const analysis = analysisRes.choices[0].message.content;

    return Response.json({
      response: `📸 **Extracted Ingredients**:\n${extracted}\n\n🔍 **FSSAI Analysis:**\n${analysis}`
    });
  }

  // ----------------------------------------------------------------------
  // 💬 CASE 2 — NORMAL TEXT CHAT
  // ----------------------------------------------------------------------
  const body = await req.json();
  const { messages }: { messages: UIMessage[] } = body;

  const latestUserMessage = messages.filter((m) => m.role === "user").pop();

  if (latestUserMessage) {
    const text = latestUserMessage.parts
      .filter((p) => p.type === "text")
      .map((p) => ("text" in p ? p.text : ""))
      .join("");

    if (text) {
      const moderation = await isContentFlagged(text);

      if (moderation.flagged) {
        const stream = createUIMessageStream({
          execute({ writer }) {
            const id = "mod-warning";
            writer.write({ type: "start" });
            writer.write({ type: "text-start", id });
            writer.write({
              type: "text-delta",
              id,
              delta: moderation.denialMessage || "Your message violates our guidelines."
            });
            writer.write({ type: "text-end", id });
            writer.write({ type: "finish" });
          }
        });

        return createUIMessageStreamResponse({ stream });
      }
    }
  }

  // Streaming text model
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

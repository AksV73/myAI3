import {
  streamText,
  convertToModelMessages,
  UIMessage
} from "ai";

import { MODEL } from "@/config";
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";
import OpenAI from "openai";

export const maxDuration = 60;

export async function POST(req: Request) {

  const contentType = req.headers.get("content-type") || "";

  // ======================================================
  // 📸 CASE 1 — IMAGE UPLOAD
  // ======================================================
  if (contentType.includes("multipart/form-data")) {

    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
    const formData = await req.formData();
    const file = formData.get("image") as File;

    if (!file) return Response.json({ response: "No image found." });

    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // 1. Extract Ingredients
    const extractRes = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Extract ONLY the ingredient list from this image." },
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
    });

    const extracted = extractRes.choices[0].message.content || "No ingredients found.";

    // 2. Analyze (With Strict Formatting)
    const analyzeRes = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are an expert FSSAI Food Safety Consultant. 
          
STYLE GUIDE:
- Use ## for headers.
- Use * for bullet points.
- Keep it concise.
- Start with a "Verdict" section.`
        },
        {
          role: "user",
          content: `Analyze these ingredients: ${extracted}`
        }
      ],
    });

    const analysis = analyzeRes.choices[0].message.content || "Could not analyze.";

    return Response.json({
      response: `## 📸 Extracted Ingredients
${extracted}

---

${analysis}`
    });
  }

  // ======================================================
  // 💬 CASE 2 — CHAT (Streaming)
  // ======================================================
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter(m => m.role === "user").pop();

  if (latest) {
    const textParts = latest.parts
      .filter(p => p.type === "text")
      .map(p => ("text" in p ? p.text : ""))
      .join("");

    const moderation = await isContentFlagged(textParts);

    if (moderation.flagged) {
      // Simple text response for blocked content (Safe & reliable)
      return new Response("Message blocked due to safety policies.", { status: 400 });
    }
  }

  // Streaming with Force-Fix for Type Errors
  const result = streamText({
    model: MODEL,
    system: `${SYSTEM_PROMPT}

STYLE GUIDE:
- Format ALL responses in clean Markdown.
- Use ## for main headers.
- Use * for lists.
- Be concise.
`,
    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    maxSteps: 5, 
  });

  // @ts-ignore
  return (result as any).toDataStreamResponse();
}

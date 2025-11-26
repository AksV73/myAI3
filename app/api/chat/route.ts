// ============================================================
//  /app/api/chat/route.ts — FINAL STABLE VERSION (OpenAI v4)
// ============================================================

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

export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ============================================================
  // 📸 IMAGE MODE
  // ============================================================
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const fd = await req.formData();
    const file = fd.get("image") as File | null;

    if (!file) {
      return Response.json({ response: "No image uploaded." });
    }

    // Convert to base64
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // ------------------------------------------------------------
    // 1) OCR EXTRACTION
    // ------------------------------------------------------------
    const ocr = await client.responses.create({
      model: "gpt-4o-mini",
      input: `
Extract ONLY the ingredient list from this food label.
Rules:
- Return a CLEAN comma-separated list.
- Include "Contains" or "May contain" if present.
- If nothing resembles ingredients, return "NOT_FOUND".

<image>${dataUrl}</image>
`
    });

    const extracted = ocr.output_text?.trim() || "NOT_FOUND";
    if (extracted === "NOT_FOUND") {
      return Response.json({ response: "⚠️ Could not detect ingredients." });
    }

    // ------------------------------------------------------------
    // 2) SAFETY ANALYSIS (STRICT JSON VIA PROMPT)
    // ------------------------------------------------------------
    const analysis = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You MUST return ONLY valid JSON. No markdown. No commentary.

JSON SCHEMA:
{
  "ingredients": [
    { "name": "", "status": "", "reason": "" }
  ],
  "summary": "",
  "overall_score": 0
}

Allowed "status" values:
- safe
- caution
- harmful
- banned
- kid-sensitive

Analyze the following ingredients using FSSAI guidelines:

${extracted}

Return ONLY the JSON object.
`
    });

    let parsed;
    try {
      parsed = JSON.parse(analysis.output_text || "{}");
    } catch (err) {
      return Response.json({
        response: "⚠️ Safety evaluation failed. Try another image."
      });
    }

    // ------------------------------------------------------------
    // 3) PRETTIFIED OUTPUT
    // ------------------------------------------------------------
    const rows = parsed.ingredients
      .map((i: any) => {
        const emoji =
          i.status === "safe"
            ? "🟢"
            : i.status === "caution"
            ? "🟡"
            : i.status === "kid-sensitive"
            ? "👶"
            : i.status === "harmful"
            ? "🔴"
            : "⛔";

        return `| ${i.name} | ${emoji} ${i.status.toUpperCase()} | ${i.reason} |`;
      })
      .join("\n");

    const table = `
| Ingredient | Status | Reason |
|-----------|--------|--------|
${rows}
`;

    return Response.json({
      response: `
📸 **Extracted Ingredients**  
${extracted}

🧪 **FSSAI Safety Evaluation**  
${table}

⭐ **Summary:**  
${parsed.summary}

📊 **Overall Score:** ${parsed.overall_score}/10
`
    });
  }

  // ============================================================
  // 💬 TEXT CHAT MODE
  // ============================================================
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter(m => m.role === "user").pop();

  if (latest) {
    const content =
      latest.parts
        .filter(p => p.type === "text")
        .map((p: any) => p.text)
        .join("") || "";

    const moderation = await isContentFlagged(content);
    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id: "blocked" });
          writer.write({
            type: "text-delta",
            id: "blocked",
            delta: moderation.denialMessage
          });
          writer.write({ type: "text-end", id: "blocked" });
          writer.write({ type: "finish" });
        }
      });

      return createUIMessageStreamResponse({ stream });
    }
  }

  // Normal chat mode
  const result = streamText({
    model: MODEL,
    system: SYSTEM_PROMPT,
    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    stopWhen: stepCountIs(10)
  });

  return result.toUIMessageStreamResponse({ sendReasoning: true });
}

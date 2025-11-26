// =====================================================
//  /api/chat/route.ts — FIXED VERSION (NO response_format)
// =====================================================

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
    const file = fd.get("image") as File;
    if (!file) return Response.json({ response: "No image uploaded." });

    // Convert to base64
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // ------------------------------------------------------------
    // 1) OCR extraction
    // ------------------------------------------------------------
    const extract = await client.responses.create({
      model: "gpt-4o-mini",
      input: `
Extract ONLY the food ingredient list from this image.
Rules:
- Return ingredients in comma-separated form.
- If nothing looks like ingredients, return "NOT_FOUND".

<image>${dataUrl}</image>
`
    });

    const extracted = extract.output_text?.trim() || "NOT_FOUND";
    if (extracted === "NOT_FOUND") {
      return Response.json({ response: "⚠️ Could not detect ingredients." });
    }

    // ------------------------------------------------------------
    // 2) Analysis — JSON enforced by prompt (NOT response_format)
    // ------------------------------------------------------------
    const analyze = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You MUST output STRICT JSON ONLY.

JSON STRUCTURE TO FOLLOW EXACTLY:
{
  "ingredients": [
    { "name": "", "status": "", "reason": "" }
  ],
  "summary": "",
  "overall_score": 0
}

Rules:
- status must be exactly one of:
  "safe", "caution", "harmful", "banned", "kid-sensitive"
- No comments outside JSON.
- No markdown.
- No explanation.

Analyze ingredients using FSSAI logic:

${extracted}

Output ONLY the JSON, nothing else.
`
    });

    let parsed;
    try {
      parsed = JSON.parse(analyze.output_text || "{}");
    } catch {
      return Response.json({
        response: "⚠️ Failed to parse results. Try another image."
      });
    }

    // ------------------------------------------------------------
    // 3) PRETTY TABLE OUTPUT
    // ------------------------------------------------------------
    const tableRows = parsed.ingredients
      .map((ing: any) => {
        const color =
          ing.status === "safe"
            ? "🟢"
            : ing.status === "caution"
            ? "🟡"
            : ing.status === "kid-sensitive"
            ? "👶"
            : ing.status === "harmful"
            ? "🔴"
            : "⛔";
        return `| ${ing.name} | ${color} ${ing.status.toUpperCase()} | ${ing.reason} |`;
      })
      .join("\n");

    const table = `
| Ingredient | Status | Reason |
|-----------|--------|--------|
${tableRows}
`;

    const finalText = `
📸 **Extracted Ingredients:**  
${extracted}

🧪 **FSSAI Safety Evaluation (India)**  
${table}

⭐ **Summary:**  
${parsed.summary}

📊 **Overall Score:** ${parsed.overall_score}/10
`;

    return Response.json({ response: finalText });
  }

  // ============================================================
  // 💬 NORMAL TEXT CHAT MODE
  // ============================================================
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter(m => m.role === "user").pop();

  if (latest) {
    const textParts =
      latest.parts
        .filter(p => p.type === "text")
        .map((p: any) => p.text)
        .join("") || "";

    const moderation = await isContentFlagged(textParts);
    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          const id = "blocked";
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id });
          writer.write({
            type: "text-delta",
            id,
            delta: moderation.denialMessage
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
    tools: { webSearch, vectorDatabaseSearch },
    stopWhen: stepCountIs(10)
  });

  return result.toUIMessageStreamResponse({ sendReasoning: true });
}

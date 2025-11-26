// ============================================================
//  /app/api/chat/route.ts — FINAL WORKING VERSION
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
  //  📸 IMAGE MODE
  // ============================================================
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY!
    });

    const fd = await req.formData();
    const file = fd.get("image") as File | null;

    if (!file) {
      return Response.json({ response: "No image uploaded." });
    }

    // Convert to base64
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // ------------------------------------------------------------
    // 1) OCR — Extract ingredients
    // ------------------------------------------------------------
    const ocr = await client.responses.create({
      model: "gpt-4o-mini",
      input: `
Extract ONLY the ingredient list from this food label.
Rules:
- Clean comma-separated ingredients only.
- If "Contains" or "May contain" exist, include them.
- Do not add commentary.
- If nothing resembles ingredients, return "NOT_FOUND".

<image>${dataUrl}</image>
`
    });

    const extracted = ocr.output_text?.trim() || "NOT_FOUND";
    if (extracted === "NOT_FOUND") {
      return Response.json({ response: "⚠️ Could not detect ingredients." });
    }

    // ------------------------------------------------------------
    // 2) SAFETY ANALYSIS — JSON via prompt
    // ------------------------------------------------------------
    const analysis = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
You MUST return ONLY valid JSON. No markdown. NO comments.

JSON STRUCTURE:
{
  "ingredients": [
    { "name": "", "status": "", "reason": "" }
  ],
  "summary": "",
  "overall_score": 0
}

Allowed statuses:
- safe
- caution
- harmful
- banned
- kid-sensitive

Analyze these ingredients using FSSAI guidelines:

${extracted}

Return ONLY THE JSON OBJECT.
`
    });

    let parsed;
    try {
      parsed = JSON.parse(analysis.output_text || "{}");
    } catch (err) {
      return Response.json({
        response: "⚠️ Could not format the safety results. Try another image."
      });
    }

    // ------------------------------------------------------------
    // 3) BEAUTIFUL TABLE OUTPUT
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
  //  💬 TEXT CHAT MODE
  // ============================================================
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter((m) => m.role === "user").pop();

  // Moderation check
  if (latest) {
    const content =
      latest.parts
        .filter((p) => p.type === "text")
        .map((

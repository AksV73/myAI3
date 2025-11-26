// ============================================================
//  /app/api/chat/route.ts — FINAL WORKING & STABLE
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

// Required for file uploads
export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ========================================================================
  //  📸 IMAGE MODE (Multipart form-data)
  // ========================================================================
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY!
    });

    const fd = await req.formData();
    const file = fd.get("image") as File | null;

    if (!file) {
      return Response.json({ response: "⚠️ No image uploaded." });
    }

    // -- Convert File → Base64 ---------------------------------------------
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // ======================================================================
    // 1) OCR — Robust Ingredient Extraction
    // ======================================================================
    const ocr = await client.responses.create({
      model: "gpt-4o-mini",
      input: `
You are an OCR expert. Extract the FULL ingredient section from this food label.

CRITICAL RULES:
- ALWAYS try your best. If blurry, extract what is readable.
- Return comma-separated ingredients.
- If "CONTAINS" or "MAY CONTAIN" appear, include them.
- Do NOT return "NOT_FOUND" unless there is ZERO readable text.
- Do NOT add commentary.
- Only output ingredient text.

<image>${dataUrl}</image>
`
    });

    const extractedRaw = ocr.output_text?.trim() || "";
    if (!extractedRaw || extractedRaw.length < 3) {
      return Response.json({
        response: "⚠️ Could not detect ingredients. Try a clearer image."
      });
    }

    const extracted = extractedRaw.replace(/\n+/g, " ").trim();

    // ======================================================================
    // 2) SAFETY ANALYSIS (GPT returns JSON by prompt discipline)
    // ======================================================================
    const analysis = await client.responses.create({
      model: "gpt-4.1-mini",
      input: `
Return ONLY valid JSON. No markdown. No commentary.

JSON STRUCTURE:
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

Analyze the following ingredients using Indian FSSAI rules:

${extracted}

Return ONLY JSON.
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

    // ======================================================================
    // 3) BEAUTIFUL TABLE OUTPUT
    // ======================================================================
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

    // ======================================================================
    // 4) FINAL RESPONSE
    // ======================================================================
    return Response.json({
      response: `
📸 **Extracted Ingredients:**  
${extracted}

🧪 **FSSAI Safety Evaluation (India)**  
${table}

⭐ **Summary:**  
${parsed.summary}

📊 **Overall Score:** ${parsed.overall_score}/10
`
    });
  }

  // ========================================================================
  //  💬 NORMAL TEXT CHAT MODE
  // ========================================================================
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter((m) => m.role === "user").pop();

  // Moderation
  if (latest) {
    const content =
      latest.parts
        .filter((p) => p.type === "text")
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
            delta: moderation.denialMessage || "Message blocked."
          });
          writer.write({ type: "text-end", id: "blocked" });
          writer.write({ type: "finish" });
        }
      });

      return createUIMessageStreamResponse({ stream });
    }
  }

  // Normal chat
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

// =====================================================
//  /api/chat/route.ts — FULL WORKING VERSION
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

// Force Node runtime
export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ========================================================================
  // 📸 CASE 1: IMAGE MODE
  // ========================================================================
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return Response.json({ response: "No image uploaded." });
    }

    // Convert to base64
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // ----------------------------------------------------------------------
    // 1) OCR — extract ingredient list
    // ----------------------------------------------------------------------
    const extractRes = await client.responses.create({
      model: "gpt-4o-mini",
      input: `
You are an OCR expert.

Extract ONLY the ingredient list from the food label image.
Rules:
- If "Ingredients:" exists → extract everything after it.
- Also extract "Contains:" or "May contain:".
- Return a CLEAN, comma-separated list.
- No extra text. No commentary.
- If nothing looks like ingredients → return "NOT_FOUND".

<image>${dataUrl}</image>
`
    });

    const extracted = extractRes.output_text?.trim() || "NOT_FOUND";

    if (!extracted || extracted === "NOT_FOUND") {
      return Response.json({
        response: "⚠️ Could not detect ingredients. Try another image."
      });
    }

    // ----------------------------------------------------------------------
    // 2) SAFETY ANALYSIS (STRICT JSON OUTPUT)
    // ----------------------------------------------------------------------
    const analyzeRes = await client.responses.create({
      model: "gpt-4.1-mini",
      response_format: { type: "json_object" }, // ensures valid JSON

      input: `
You MUST return a valid JSON object matching EXACTLY the structure below:

{
  "ingredients": [
    {
      "name": "",
      "status": "safe | caution | harmful | banned | kid-sensitive",
      "reason": ""
    }
  ],
  "summary": "",
  "overall_score": 0
}

Analyze this ingredient list using FSSAI rules:

${extracted}

Return ONLY the JSON object.
`
    });

    let parsed;
    try {
      parsed = JSON.parse(analyzeRes.output_text || "{}");
    } catch (e) {
      return Response.json({
        response: "⚠️ Could not format safety results. Try another image."
      });
    }

    // ----------------------------------------------------------------------
    // 3) Pretty formatting for UI
    // ----------------------------------------------------------------------

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

    const finalResponse = `
📸 **Extracted Ingredients:**  
${extracted}

🧪 **FSSAI Safety Evaluation (India)**  
${table}

⭐ **Summary:**  
${parsed.summary}

📊 **Overall Score:** ${parsed.overall_score}/10
`;

    return Response.json({ response: finalResponse });
  }

  // ========================================================================
  // 💬 CASE 2: TEXT CHAT MODE
  // ========================================================================
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();
  if (latest) {
    const textParts =
      latest.parts
        .filter((p) => p.type === "text")
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
            delta:
              moderation.denialMessage || "Your message violated policy."
          });
          writer.write({ type: "text-end", id });
          writer.write({ type: "finish" });
        }
      });

      return createUIMessageStreamResponse({ stream });
    }
  }

  // Normal streaming chat
  const result = streamText({
    model: MODEL,
    system: SYSTEM_PROMPT,
    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    stopWhen: stepCountIs(10)
  });

  return result.toUIMessageStreamResponse({ sendReasoning: true });
}

// ============================================================
//  /app/api/chat/route.ts — CLEAN + CRISP OUTPUT VERSION
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

export const maxDuration = 30;

// ------------------------------------------------------------
// 🛠️ Helper Functions For Cleanup + Tables
// ------------------------------------------------------------

function extractByKeyword(text: string, keyword: string) {
  return text
    .split(" - ")
    .filter((line) => line.toLowerCase().includes(keyword))
    .map((line) => line.split(" ")[0].trim())
    .filter(Boolean);
}

function extractSafe(text: string) {
  return extractByKeyword(text, "safe");
}

function extractCaution(text: string) {
  return extractByKeyword(text, "caution");
}

function extractHarmful(text: string) {
  return extractByKeyword(text, "harmful");
}

function extractBanned(text: string) {
  return extractByKeyword(text, "banned");
}

function extractKid(text: string) {
  return extractByKeyword(text, "kid");
}

function formatIngredientsAsTable(text: string) {
  const rows = text
    .split(" - ")
    .filter(
      (line) =>
        line.includes("Safe") ||
        line.includes("Harmful") ||
        line.includes("Banned") ||
        line.includes("Caution") ||
        line.includes("Kid")
    )
    .map((line) => {
      const ing = line.split(" ")[0].trim();
      let status = "🟢 Safe";

      if (line.includes("Harmful")) status = "🔴 Harmful";
      else if (line.includes("Banned")) status = "⛔ Banned";
      else if (line.includes("Caution")) status = "🟡 Caution";
      else if (line.toLowerCase().includes("kid")) status = "👶 Kid-sensitive";

      const reason =
        line.includes("(") && line.includes(")")
          ? line.substring(line.indexOf("(") + 1, line.indexOf(")"))
          : "—";

      return `| ${ing} | ${status} | ${reason} |`;
    })
    .join("\n");

  return `
| Ingredient | Status | Notes |
|-----------|--------|--------|
${rows}
`;
}

// ------------------------------------------------------------
//  📌 MAIN ROUTE
// ------------------------------------------------------------
export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ============================================================
  //  📸 IMAGE MODE
  // ============================================================
  if (contentType.includes("multipart/form-data")) {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const fd = await req.formData();
    const file = fd.get("image") as File | null;

    if (!file) {
      return Response.json({ response: "No image uploaded." });
    }

    // Convert image → Base64
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // ------------------------------------------------------------
    //  OCR — Extract ingredients
    // ------------------------------------------------------------
    const ocr = await client.responses.create({
      model: "gpt-4-1-mini",
      input: `
Extract ONLY the ingredient list from this food label.
Rules:
- Clean comma-separated ingredients only.
- Include "Contains" or "May contain" if present.
- No commentary.
- If nothing resembles ingredients, return "NOT_FOUND".

<image>${dataUrl}</image>
`
    });

    const extracted = ocr.output_text?.trim() || "NOT_FOUND";
    if (extracted === "NOT_FOUND") {
      return Response.json({
        response: "⚠️ Could not detect ingredients. Try a clearer image."
      });
    }

    // ------------------------------------------------------------
    //  FSSAI Analysis (free-form, but we format later)
    // ------------------------------------------------------------
    const analysis = await client.responses.create({
      model: "gpt-4-1-mini",
      input: `
You are an Indian FSSAI Safety Analyzer.

Given these ingredients:

${extracted}

Classify each ingredient into:
- Safe
- Caution
- Harmful
- Banned
- Kid-sensitive

Return bullet points EXACTLY like:
- Water 🟢 Safe (harmless)
- Soybean Oil ⛔ Banned (FSSAI trans fat rule)
- Milk 👶 Kid-sensitive (allergen)

Do NOT add extra sections. Just the list.
`
    });

    const text = analysis.output_text || "";

    // ------------------------------------------------------------
    //  CLEAN & CRISP OUTPUT
    // ------------------------------------------------------------

    const finalResponse = `
📸 **Extracted Ingredients**  
${extracted}

---

### 🧪 FSSAI Safety Summary

| Category | Items |
|---------|--------|
| 🟢 Safe | ${extractSafe(text).join(", ") || "—"} |
| 🟡 Caution | ${extractCaution(text).join(", ") || "—"} |
| 🔴 Harmful | ${extractHarmful(text).join(", ") || "—"} |
| ⛔ Banned | ${extractBanned(text).join(", ") || "—"} |
| 👶 Kid-Sensitive | ${extractKid(text).join(", ") || "—"} |

---

### 📋 Detailed Ingredient Table
${formatIngredientsAsTable(text)}

---

If you want allergen mapping, preservatives check, or kid safety score, just ask!
`;

    return Response.json({ response: finalResponse });
  }

  // ============================================================
  //  💬 NORMAL CHAT MODE
  // ============================================================
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

  // Normal chat flow
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

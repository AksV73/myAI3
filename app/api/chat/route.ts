// ============================================================
//  FINAL STABLE ROUTE — ZERO TYPE ERRORS
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

// -----------------------------------------
// HELPERS
// -----------------------------------------
function safe(value: any, fallback = ""): string {
  return value ? String(value) : fallback;
}

function makeTable(safetyLines: string[]) {
  if (!safetyLines.length) return "No data.";

  const rows = safetyLines
    .map((line) => {
      const match = line.match(/^(.*?)\s+(🟢|🟡|🔴|⛔|👶)\s+([^(]+)(?:\((.*?)\))?$/);
      if (!match) return null;

      const name = match[1].trim();
      const icon = match[2].trim();
      const status = match[3].trim();
      const reason = match[4] ? match[4].trim() : "—";

      return `| ${name} | ${icon} ${status} | ${reason} |`;
    })
    .filter(Boolean)
    .join("\n");

  return `
| Ingredient | Status | Reason |
|-----------|--------|--------|
${rows}
`;
}

// ============================================================
//  MAIN ROUTE
// ============================================================
export async function POST(req: Request) {
  const type = req.headers.get("content-type") || "";

  // ============================================================
  // IMAGE UPLOAD MODE
  // ============================================================
  if (type.includes("multipart/form-data")) {
    try {
      const OpenAI = (await import("openai")).default;
      const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

      const fd = await req.formData();
      const file = fd.get("image") as File | null;

      if (!file) {
        return Response.json({ response: "No file uploaded." });
      }

      // convert to base64
      const buffer = Buffer.from(await file.arrayBuffer());
      const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

      // STEP 1 — OCR
      const ocrRes = await client.responses.create({
        model: "gpt-4-1-mini",
        input: `
Extract ONLY the ingredient list from this food label.
If not found return: NOT_FOUND
<image>${dataUrl}</image>
`
      });

      const extracted = safe(ocrRes.output_text).trim();

      if (!extracted || extracted === "NOT_FOUND") {
        return Response.json({
          response: "⚠️ Could not detect ingredients. Try a clearer image."
        });
      }

      // STEP 2 — FSSAI Safety Analysis
      const safetyRes = await client.responses.create({
        model: "gpt-4-1-mini",
        input: `
Classify each ingredient from this list:

${extracted}

Return ONLY bullet points like:
- Sugar 🟢 Safe (reason)
- Milk 👶 Kid-sensitive (allergen)
- XYZ ⛔ Banned (FSSAI rule)
`
      });

      const safetyText = safe(safetyRes.output_text);
      const lines = safetyText
        .split("\n")
        .map((x) => x.replace(/^- /, "").trim())
        .filter((x) => x.length > 0);

      const table = makeTable(lines);

      return Response.json({
        response: `
📸 **Extracted Ingredients**
${extracted}

### 🧪 FSSAI Safety Table
${table}

Need allergen or kid safety score? Ask me!
`
      });
    } catch (err: any) {
      return Response.json({ response: `❌ Server error: ${err?.message}` });
    }
  }

  // ============================================================
  // CHAT MODE
  // ============================================================
  const { messages }: { messages: UIMessage[] } = await req.json();
  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const text =
      latest.parts
        ?.filter((p) => p.type === "text")
        ?.map((p: any) => p.text)
        ?.join("") || "";

    const moderation = await isContentFlagged(text);

    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id: "blocked" });
          writer.write({
            type: "text-delta",
            id: "blocked",
            delta: moderation.denialMessage ?? "Message blocked."
          });
          writer.write({ type: "text-end", id: "blocked" });
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

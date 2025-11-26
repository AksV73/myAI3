// ===============================================
// /app/api/chat/route.ts — BEAUTIFIED OUTPUT
// ===============================================

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

    // ======================================================
    // 📸 CASE 1 — IMAGE UPLOAD
    // ======================================================
    if (contentType.includes("multipart/form-data")) {
        const OpenAI = (await import("openai")).default;
        const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

        const formData = await req.formData();
        const file = formData.get("image") as File;

        if (!file) {
            return Response.json({ response: "No image found." });
        }

        // Convert uploaded image → Base64 data URL
        const buffer = Buffer.from(await file.arrayBuffer());
        const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

        // ======================================================
        // 🔍 1) Extract Ingredient List
        // ======================================================
        const extractRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
You are a food-label OCR expert.

Extract ONLY:
1. Ingredient list  
2. Contains list / May contain allergens  

Return in clean comma-separated format.

<image>${dataUrl}</image>
`
        });

        const extracted = extractRes.output_text || "No ingredients found.";

        // ======================================================
        // 🧪 2) Analyze using FSSAI safety rules
        // ======================================================
        const analyzeRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
Classify the following ingredients based on FSSAI safety guidelines:

${extracted}

For EACH ingredient return JSON objects ONLY in this format:

{
  "ingredient": "",
  "status": "safe | caution | harmful | banned | kid-sensitive",
  "reason": ""
}

After this array, include:

{
  "summary": "",
  "overall_score": 0
}

DO NOT return any explanation outside of JSON.
`
        });

        let parsed;
        try {
            parsed = JSON.parse(analyzeRes.output_text || "{}");
        } catch {
            return Response.json({
                response: "Could not format the safety results. Try another image."
            });
        }

        const rows = Array.isArray(parsed) ? parsed.slice(0, -1) : [];
        const summaryObj = Array.isArray(parsed) ? parsed[parsed.length - 1] : {};

        // ======================================================
        // 🎨 BEAUTIFY INTO A TABLE
        // ======================================================
        const emojiMap: Record<string, string> = {
            "safe": "🟢 Safe",
            "caution": "🟡 Caution",
            "harmful": "🔴 Harmful",
            "banned": "⛔ Banned",
            "kid-sensitive": "👶 Kid Sensitive"
        };

        const table = `
| Ingredient | Status | Notes |
|-----------|--------|-------|
${rows
    .map(
        (r: any) =>
            `| ${r.ingredient} | ${emojiMap[r.status] || r.status} | ${r.reason} |`
    )
    .join("\n")}
`;

        // ======================================================
        // 🧼 Final Pretty Output
        // ======================================================
        const finalResponse = `
## 📸 Extracted Ingredients
${extracted}

---

## 🧪 FSSAI Safety Table
${table}

---

## 📝 Summary  
${summaryObj.summary || ""}

### ⭐ Overall Safety Score: **${summaryObj.overall_score ?? "-"} / 10**
`;

        return Response.json({ response: finalResponse });
    }

    // ======================================================
    // 💬 CASE 2 — STANDARD CHAT MODE
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
            const stream = createUIMessageStream({
                execute({ writer }) {
                    const id = "blocked";
                    writer.write({ type: "start" });
                    writer.write({ type: "text-start", id });
                    writer.write({
                        type: "text-delta",
                        id,
                        delta: moderation.denialMessage || "Message blocked."
                    });
                    writer.write({ type: "text-end", id });
                    writer.write({ type: "finish" });
                }
            });

            return createUIMessageStreamResponse({ stream });
        }
    }

    // NORMAL CHAT
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

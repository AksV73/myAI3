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
        // 🔍 1) Extract Ingredient List INTELLIGENTLY
        // ======================================================
        const extractRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
You are a food-label OCR expert.

From the image below, extract the INGREDIENT LIST if it exists.

Rules:
- Return ONLY the ingredient list (comma or line separated).
- Ignore nutrition tables, calories, barcode text, slogans.
- If the label has "CONTAINS" or "MAY CONTAIN", ALSO extract them.
- If no explicit "Ingredients:" header exists, detect ingredient-like text.
- Remove non-food text.
- Clean and normalize the output.

If nothing resembles ingredients, return: "No ingredients found."

<image>${dataUrl}</image>
`
        });

        const extracted = extractRes.output_text || "No ingredients found.";

        // ======================================================
        // 🧪 2) Analyze Using FSSAI Logic
        // ======================================================
        const analyzeRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
You are an Indian FSSAI Additive Analyzer.

Given these extracted ingredients:

${extracted}

Classify each into:
- SAFE  
- HARMFUL  
- BANNED (FSSAI)  
- KID-SENSITIVE  

Return results in clear bullet points with emojis:
🟢 Safe  
🟡 Caution  
🔴 Harmful  
⛔ Banned  
👶 Kid-sensitive  
`
        });

        const analysis = analyzeRes.output_text || "Could not analyze ingredients.";

        return Response.json({
            response:
`📸 **Extracted Ingredients:**  
${extracted}

🔍 **FSSAI Safety Analysis:**  
${analysis}`
        });
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

    // ======================================================
    // 🤖 Normal Streaming Chat
    // ======================================================
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

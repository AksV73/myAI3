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

    // ---------------------
    // 📸 CASE 1 — IMAGE UPLOAD
    // ---------------------
    if (contentType.includes("multipart/form-data")) {
        const OpenAI = (await import("openai")).default;
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

        const formData = await req.formData();
        const file = formData.get("image") as File;

        if (!file) {
            return Response.json({ response: "No image received." });
        }

        // Convert file → Base64 DataURL
        const buffer = Buffer.from(await file.arrayBuffer());
        const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

        // 1️⃣ OCR extraction using <image> tag
        const visionRes = await openai.responses.create({
            model: "gpt-4.1-mini",
            input: `Extract ONLY the ingredient list from this food label image:
<image>${dataUrl}</image>`
        });

        const extracted = visionRes.output_text || "Could not extract ingredients.";

        // 2️⃣ FSSAI risk analysis
        const analysisRes = await openai.responses.create({
            model: "gpt-4.1-mini",
            input:
                `You are an Indian FSSAI additive checker. 
Classify each ingredient as SAFE / HARMFUL / BANNED / KID-SENSITIVE.
Use bullet points.

Ingredients:
${extracted}`
        });

        const analysis = analysisRes.output_text;

        return Response.json({
            response:
`📸 **Extracted Ingredients:**  
${extracted}

🔍 **FSSAI Safety Analysis:**  
${analysis}`
        });
    }

    // ---------------------
    // 💬 CASE 2 — NORMAL CHAT
    // ---------------------
    const { messages }: { messages: UIMessage[] } = await req.json();

    const latestUserMessage = messages.filter(m => m.role === "user").pop();

    if (latestUserMessage) {
        const textParts = latestUserMessage.parts
            .filter(p => p.type === "text")
            .map(p => ("text" in p ? p.text : ""))
            .join("");

        if (textParts) {
            const moderation = await isContentFlagged(textParts);

            if (moderation.flagged) {
                const stream = createUIMessageStream({
                    execute({ writer }) {
                        const id = "mod-warning";
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
    }

    // Normal RAG chat
    const result = streamText({
        model: MODEL,
        system: SYSTEM_PROMPT,
        messages: convertToModelMessages(messages),
        tools: {
            webSearch,
            vectorDatabaseSearch
        },
        stopWhen: stepCountIs(10)
    });

    return result.toUIMessageStreamResponse({
        sendReasoning: true
    });
}


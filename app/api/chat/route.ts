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
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

        const formData = await req.formData();
        const file = formData.get("image") as File;

        if (!file) {
            return Response.json({ response: "No file uploaded." });
        }

        // convert to base64 dataUrl
        const arrayBuffer = await file.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString("base64");
        const dataUrl = `data:${file.type};base64,${base64}`;

        // 1️⃣ EXTRACT INGREDIENT TEXT USING GPT-4o-mini
        const visionRes = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content: "Extract ONLY the ingredient list from the food label. Plain text only."
                },
                {
                    role: "user",
                    content: [
                        { type: "text", text: "Extract ingredients from this image:" },
                        { type: "image", image: dataUrl }
                    ]
                }
            ]
        });

        const extracted = visionRes.choices[0].message.content || "Unable to extract ingredients.";

        // 2️⃣ ANALYZE INGREDIENTS
        const analysisRes = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content:
                        "You are an Indian FSSAI Additive Analyzer. Classify ingredients into SAFE / HARMFUL / BANNED / KID-SENSITIVE. Use bullet points."
                },
                {
                    role: "user",
                    content: extracted
                }
            ]
        });

        const analysis = analysisRes.choices[0].message.content || "No analysis available.";

        return Response.json({
            response:
                `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
                `🔍 **FSSAI Safety Analysis:**\n${analysis}`
        });
    }

    // ======================================================
    // 💬 CASE 2 — NORMAL CHAT
    // ======================================================

    const { messages }: { messages: UIMessage[] } = await req.json();

    const latestUserMessage = messages.filter(m => m.role === "user").pop();

    if (latestUserMessage) {
        const textParts = latestUserMessage.parts
            .filter(p => p.type === "text")
            .map(p => ("text" in p ? p.text : ""))
            .join("");

        if (textParts) {
            const moderationResult = await isContentFlagged(textParts);

            if (moderationResult.flagged) {
                const stream = createUIMessageStream({
                    execute({ writer }) {
                        const id = "moderation-warning";

                        writer.write({ type: "start" });
                        writer.write({ type: "text-start", id });
                        writer.write({
                            type: "text-delta",
                            id,
                            delta:
                                moderationResult.denialMessage ||
                                "Your message violates our guidelines."
                        });
                        writer.write({ type: "text-end", id });
                        writer.write({ type: "finish" });
                    }
                });

                return createUIMessageStreamResponse({ stream });
            }
        }
    }

    // Normal streaming chat from your RAG
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

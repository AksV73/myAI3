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

// ---------------------------------------------------------
// 🆕 Markdown Beautifier
// ---------------------------------------------------------
function beautifyMarkdown(text: string): string {
    return text
        .replace(/\. /g, ".\n\n")      // add spacing after sentences
        .replace(/###/g, "\n###")      // add line breaks before headings
        .replace(/\n{3,}/g, "\n\n")    // normalize excessive newlines
        .trim();
}

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

Extract ONLY the ingredient list from this image:

<image>${dataUrl}</image>
`
        });

        const extracted = extractRes.output_text || "No ingredients found.";

        // ======================================================
        // 🧪 2) Ingredient Safety Analysis
        // ======================================================
        const analyzeRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
You are an Indian FSSAI Additive Analyzer.

Analyze these ingredients:

${extracted}

Return results in:
- Clean Markdown
- Bullet points
- Emojis (🟢🟡🔴⛔👶)
`
        });

        const analysis = beautifyMarkdown(analyzeRes.output_text || "Could not analyze ingredients.");

        return Response.json({
            response:
`## 📸 Extracted Ingredients  
${beautifyMarkdown(extracted)}

## 🔍 FSSAI Safety Analysis  
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
    // 🤖 Streaming Chat with Markdown Formatting
    // ======================================================
    const result = streamText({
        model: MODEL,

        system:
`${SYSTEM_PROMPT}

FORMAT ALL ANSWERS IN CLEAN MARKDOWN.
Use:
- Headings (##)
- Bullet points
- Short paragraphs
- Bold important concepts
No long dense paragraphs.
`,

        messages: convertToModelMessages(messages),
        tools: { webSearch, vectorDatabaseSearch },
        stopWhen: stepCountIs(10)
    });

    // -----------------------------------------------------
    // 🆕 Transform stream output to beautify markdown
    // -----------------------------------------------------
    const stream = result.toDataStream();

    const transformed = stream.experimental_transform({
        transform(chunk) {
            if (chunk.type === "text-delta" && chunk.text) {
                chunk.text = beautifyMarkdown(chunk.text);
            }
            return chunk;
        }
    });

    return transformed.toResponse();
}

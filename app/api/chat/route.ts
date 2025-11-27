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
// 🆕 ADD THIS — Markdown Beautifier
// ---------------------------------------------------------
function beautifyMarkdown(text: string): string {
    return text
        .replace(/\. /g, ".\n\n")      // spacing after sentences
        .replace(/###/g, "\n###")      // spacing before headings
        .replace(/\n{3,}/g, "\n\n")    // normalize extra newlines
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
        // 🔍 1) Extract Ingredients
        // ======================================================
        const extractRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
You are a food-label OCR expert.
Return ONLY the extracted ingredient list in clean format.

<image>${dataUrl}</image>
`
        });

        const extracted = extractRes.output_text || "No ingredients found.";

        // ======================================================
        // 🧪 2) FSSAI Safety Analysis
        // ======================================================
        const analyzeRes = await client.responses.create({
            model: "gpt-4.1-mini",
            input: `
You are an Indian FSSAI Additive Analyzer.

Given these extracted ingredients:

${extracted}

Return the analysis in:
- Clean Markdown
- Bullet points
- Emojis for safety level
`
        });

        const analysis = beautifyMarkdown(analyzeRes.output_text || "Could not analyze ingredients.");

        // 🆕 PRETTIFIED IMAGE OUTPUT
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
    // 🤖 Normal Streaming Chat
    // ======================================================
    const result = streamText({
        model: MODEL,

        // 🆕 Improve formatting in AI behavior
        system:
`${SYSTEM_PROMPT}

FORMAT ALL ANSWERS IN CLEAN MARKDOWN.
Use:
- Headings (##)
- Bullet points
- Short paragraphs
- Bold important concepts
Avoid long dense paragraphs.
`,

        messages: convertToModelMessages(messages),
        tools: { webSearch, vectorDatabaseSearch },
        stopWhen: stepCountIs(10)
    });

    // -----------------------------------------------------
    // 🆕 Beautify the streamed result before sending
    // -----------------------------------------------------
    return result.toUIMessageStreamResponse({
        sendReasoning: true,
        transform: (text) => beautifyMarkdown(text)  // <— MAGIC LINE
    });
}

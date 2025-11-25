
import { streamText, UIMessage, convertToModelMessages, stepCountIs, createUIMessageStream, createUIMessageStreamResponse } from 'ai';
import { MODEL } from '@/config';
import { SYSTEM_PROMPT } from '@/prompts';
import { isContentFlagged } from '@/lib/moderation';
import { webSearch } from './tools/web-search';
import { vectorDatabaseSearch } from './tools/search-vector-database';

export const maxDuration = 30;
export async function POST(req: Request) {
    
                    

    const contentType = req.headers.get("content-type") || "";

    // ======================================================
    // 📸 CASE 1: IMAGE UPLOAD
    // ======================================================
    if (contentType.includes("multipart/form-data")) {

        // --- lazy import allowed INSIDE function ---
        const OpenAI = (await import("openai")).default;
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

        const formData = await req.formData();
        const file = formData.get("image") as File;

        if (!file) {
            return Response.json({ response: "No file uploaded." });
        }

        const arrayBuffer = await file.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString("base64");

        // 1️⃣ OCR with GPT-4o-mini Vision
        const visionRes = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content: "Extract ONLY the ingredient list text from the food label. Plain text only.",
                },
                {
                    role: "user",
                    content: [
                        {
                            type: "input_image",
                            image_url: `data:${file.type};base64,${base64}`,
                        },
                    ],
                },
            ],
        });

        const extracted = visionRes.choices[0].message.content;

        // 2️⃣ Analyze ingredients
        const analysisRes = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content:
                        "You are an Indian FSSAI Additive Analyzer. Classify ingredients into SAFE / HARMFUL / BANNED / KID-SENSITIVE. Respond in clean bullet points.",
                },
                {
                    role: "user",
                    content: extracted,
                },
            ],
        });

        const analysis = analysisRes.choices[0].message.content;

        return Response.json({
            response:
                `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
                `🔍 **FSSAI Safety Analysis:**\n${analysis}`,
        });
    }

    // ======================================================
    // 💬 CASE 2: NORMAL CHAT — your original code
    // ======================================================
    const { messages }: { messages: UIMessage[] } = await req.json();

    const latestUserMessage = messages
        .filter(msg => msg.role === "user")
        .pop();

    if (latestUserMessage) {
        const textParts = latestUserMessage.parts
            .filter(part => part.type === "text")
            .map(part => ("text" in part ? part.text : ""))
            .join("");

        if (textParts) {
            const moderationResult = await isContentFlagged(textParts);

            if (moderationResult.flagged) {
                const stream = createUIMessageStream({
                    execute({ writer }) {
                        const textId = "moderation-denial-text";

                        writer.write({ type: "start" });
                        writer.write({ type: "text-start", id: textId });
                        writer.write({
                            type: "text-delta",
                            id: textId,
                            delta:
                                moderationResult.denialMessage ||
                                "Your message violates our guidelines.",
                        });
                        writer.write({ type: "text-end", id: textId });
                        writer.write({ type: "finish" });
                    },
                });

                return createUIMessageStreamResponse({ stream });
            }
        
    }

    const result = streamText({
        model: MODEL,
        system: SYSTEM_PROMPT,
        messages: convertToModelMessages(messages),
        tools: {
            webSearch,
            vectorDatabaseSearch,
        },
        stopWhen: stepCountIs(10),
    });

    return result.toUIMessageStreamResponse({
        sendReasoning: true,
    });
}




           
 

               
    }

    const result = streamText({
        model: MODEL,
        system: SYSTEM_PROMPT,
        messages: convertToModelMessages(messages),
        tools: {
            webSearch,
            vectorDatabaseSearch,
        },
        stopWhen: stepCountIs(10),
    });

    return result.toUIMessageStreamResponse({
        sendReasoning: true,
    });
}

    }

    const result = streamText({
        model: MODEL,
        system: SYSTEM_PROMPT,
        messages: convertToModelMessages(messages),
        tools: {
            webSearch,
            vectorDatabaseSearch,
        },
        stopWhen: stepCountIs(10),
        providerOptions: {
            openai: {
                reasoningSummary: 'auto',
                reasoningEffort: 'low',
                parallelToolCalls: false,
            }
        }
    });

    return result.toUIMessageStreamResponse({
        sendReasoning: true,
    });
}

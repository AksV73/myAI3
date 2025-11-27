import {
  streamText,
  UIMessage,
  convertToModelMessages,
  stepCountIs,
  createUIMessageStream,
  createUIMessageStreamResponse
} from "ai";

import { MODEL } from "@/config"; // Ensure this exports a string like 'gpt-4o'
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";
import OpenAI from "openai"; // Import OpenAI directly here to be safe

export const maxDuration = 60; // Increased duration slightly for image processing

export async function POST(req: Request) {

  const contentType = req.headers.get("content-type") || "";

  // ======================================================
  // 📸 CASE 1 — IMAGE UPLOAD (Single Response)
  // ======================================================
  if (contentType.includes("multipart/form-data")) {

    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

    const formData = await req.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return Response.json({ response: "No image found." });
    }

    // Convert uploaded image → Base64 data URL
    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // 🔍 1) Extract Ingredient List
    const extractRes = await client.chat.completions.create({
      model: "gpt-4o-mini", // FIXED MODEL NAME
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Extract ONLY the ingredient list from this image. Do not add any conversational filler." },
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
    });

    const extracted = extractRes.choices[0].message.content || "No ingredients found.";

    // 🧪 2) Ingredient Safety Analysis
    // We put the STRICT formatting instructions here
    const analyzeRes = await client.chat.completions.create({
      model: "gpt-4o", // Use the big model for better reasoning
      messages: [
        {
          role: "system",
          content: `You are an expert FSSAI Food Safety Consultant. 
Your goal is to explain food labels to a layperson.
          
CRITICAL FORMATTING RULES:
1. Use H2 headers (##) for sections. NEVER use H3 (###) or H4 (####).
2. Use standard bullet points (*).
3. Do not use bold (**) excessively. Only bold the name of the ingredient.
4. Keep descriptions short (1 sentence).
5. Add a "Verdit" section at the top with a single emoji status.
`
        },
        {
          role: "user",
          content: `Analyze these ingredients for safety, allergens, and child suitability:
          
${extracted}`
        }
      ],
    });

    const analysis = analyzeRes.choices[0].message.content || "Could not analyze ingredients.";

    // Return a clean JSON. The UI will render the markdown.
    return Response.json({
      response: `## 📸 Extracted Ingredients
${extracted}

---

${analysis}`
    });
  }

  // ======================================================
  // 💬 CASE 2 — STANDARD CHAT MODE (Streaming)
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
  // 🤖 Streaming Chat
  // ======================================================
  const result = streamText({
    model: MODEL,

    // We fix the output output HERE in the system prompt, not with Regex
    system: `${SYSTEM_PROMPT}

STYLE GUIDE:
- Format ALL responses in clean Markdown.
- Use ## for main headers.
- Use * for lists.
- Avoid large blocks of text.
- Be concise and friendly.
`,

    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    maxSteps: 10, 
  });

  // FIXED: Standard way to return a stream in Vercel AI SDK
  return result.toDataStreamResponse();
}

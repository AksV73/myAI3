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
import OpenAI from "openai"; // Ensure OpenAI is imported

// Increased duration slightly for complex vision/analysis tasks
export const maxDuration = 60; 

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

    // 1. Extract Ingredients (Use a faster, smaller model for simple OCR)
    const extractRes = await client.chat.completions.create({
      model: "gpt-4o-mini", // FIXED: Using the correct, fast model name
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

    // 2. Ingredient Safety Analysis (Use a powerful model for reasoning/formatting)
    const analyzeRes = await client.chat.completions.create({
      model: "gpt-4o", // Using a powerful model for high-quality analysis
      messages: [
        {
          // FIX: Strict Formatting Prompt applied here to solve the ugly text issue
          role: "system",
          content: `You are an expert FSSAI Food Safety Consultant. 
          
Your primary goal is to provide a clean, readable, and structured analysis for a non-technical user.
          
CRITICAL FORMATTING RULES:
1. Use H2 headers (##) for sections. NEVER use H3 (###) or H4 (####) which cause visual errors.
2. Use standard bullet points (*) for lists.
3. Do not use bold (**) excessively.
4. Keep descriptions concise (one or two sentences max).
5. Always include a section summarizing the overall verdict and one for child safety/allergens.
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
    
    // FIX: Strict Formatting Prompt added here for the chat model as well
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
    
    // COMPILATION FIX: Using the property compatible with your SDK version
    stopWhen: stepCountIs(5) 
  });

  // COMPILATION FIX: The most reliable way to return the raw stream, 
  // bypassing the conflicting Vercel SDK utility methods that caused the errors.
  return new Response(result.stream, {
      headers: { 
        "Content-Type": "text/plain", 
        "Cache-Control": "no-cache" 
      },
  });
}

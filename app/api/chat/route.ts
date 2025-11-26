// ------------------------------------------------------
// FORCE NODE RUNTIME (SHARP NEEDS NODE ENV)
// ------------------------------------------------------
export const runtime = "nodejs";
export const preferredRegion = "bom1";

import OpenAI from "openai";
import sharp from "sharp";
import { Buffer } from "buffer";

import {
  streamText,
  UIMessage,
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs
} from "ai";

import { MODEL } from "@/config";
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

export const maxDuration = 30;

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ============================================================================================
  // 📸 IMAGE MODE — OCR + ANALYSIS
  // ============================================================================================
  if (contentType.includes("multipart/form-data")) {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file)
      return Response.json({ response: "No image uploaded." }, { status: 400 });

    // Convert File → Buffer safely
    const ab = await file.arrayBuffer();
    const buffer = Buffer.from(new Uint8Array(ab));

    // optional light auto-rotate
    let enhanced = buffer;
    try {
      enhanced = await sharp(buffer).rotate().toBuffer();
    } catch {
      enhanced = buffer;
    }

    const dataUrl = `data:${file.type};base64,${enhanced.toString("base64")}`;

    // --------------------------------------
    // OCR STEP
    // --------------------------------------
    const ocrPrompt = `
Extract ONLY the cosmetic ingredient list from this image.

Rules:
- Only text after "Ingredients:"
- No guesswork
- No extra lines
- If unreadable: UNREADABLE

<image>${dataUrl}</image>
`;

    const ocrRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      input: ocrPrompt
    });

    const extracted = ocrRes.output_text?.trim() || "UNREADABLE";

    // --------------------------------------
    // ANALYSIS STEP
    // --------------------------------------
    const analysisPrompt = `
Analyze the cosmetics ingredient list below for Indian safety standards:

"${extracted}"

Return STRICT JSON only:
{
  "ingredients": [
    { "name": "", "classification": "", "reason": "" }
  ],
  "score": 0,
  "summary": ""
}

Classification rules:
- SAFE
- IRRITANT (fragrance, sulfates, EOs)
- RESTRICTED/BANNED (mercury, lead, hydroquinone OTC)
- PREGNANCY_UNSAFE (retinoids, high SA)
- KID_UNSAFE (MIT/CMIT, fragrances)
- COMEDOGENIC (coconut oil, cocoa butter, shea)
`;

    const analysisRes = await openai.responses.create({
      model: "gpt-4o-mini",
      temperature: 0,
      max_output_tokens: 1200,
      input: analysisPrompt
    });

    let parsed;
    try {
      parsed = JSON.parse(analysisRes.output_text || "{}");
    } catch {
      parsed = {
        ingredients: [],
        score: 5,
        summary: "Could not parse JSON.",
        raw: analysisRes.output_text
      };
    }

    return Response.json({ response: JSON.stringify(parsed, null, 2) });
  }

  // ============================================================================================
  // 💬 NORMAL TEXT CHAT MODE
  // ============================================================================================

  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const textParts =
      latest.parts
        ?.filter((p) => p.type === "text")
        .map((p) => ("text" in p ? p.text : ""))
        .join("") || "";

    const moderation = await isContentFlagged(textParts);

    if (moderation.flagged) {
      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id: "blocked" });
          writer.write({
            type: "text-delta",
            id: "blocked",
            delta:
              moderation.denialMessage || "Your message violates our policy."
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

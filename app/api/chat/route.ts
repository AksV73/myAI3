// app/api/chat/route.ts
export const runtime = "nodejs";

import OpenAI from "openai";
import {
  streamText,
  convertToModelMessages,
  createUIMessageStreamResponse,
  stepCountIs,
  UIMessage
} from "ai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// -------------------------------------------------------
// POST HANDLER
// -------------------------------------------------------
export async function POST(req: Request) {
  const type = req.headers.get("content-type") || "";

  // ---------------------------------------------------
  // IMAGE MODE
  // ---------------------------------------------------
  if (type.includes("multipart/form-data")) {
    const form = await req.formData();
    const file = form.get("image") as File | null;
    if (!file) {
      return Response.json({ response: "❌ No image uploaded." });
    }

    const bytes = await file.arrayBuffer();
    const base64 = Buffer.from(bytes).toString("base64");
    const dataUrl = `data:${file.type};base64,${base64}`;

    // ------------------ OCR --------------------------
    const ocr = await openai.responses.create({
      model: "gpt-4o-mini",
      input: `
Extract ONLY the "Ingredients" text from this food label image.
Do NOT guess. If unreadable, return "UNREADABLE".

<image>${dataUrl}</image>
`
    });

    const extracted = ocr.output_text?.trim() || "UNREADABLE";

    // ------------------ ANALYSIS ----------------------
    const analysis = await openai.responses.create({
      model: "gpt-4o-mini",
      input: `
You are an FSSAI food-safety expert.

Analyze this ingredient list:

"${extracted}"

Return STRICT JSON ONLY:

{
  "ingredients": [
    { "name": "", "risk": "", "reason": "" }
  ],
  "score": 0,
  "summary": ""
}

Risk levels:
- SAFE  
- CAUTION  
- UNSAFE  
- KID_UNSAFE  
- BANNED_FSSAI  
`
    });

    let parsed;
    try {
      parsed = JSON.parse(analysis.output_text || "{}");
    } catch {
      parsed = {
        ingredients: [],
        score: 5,
        summary: "Could not parse JSON.",
        raw: analysis.output_text
      };
    }

    return Response.json({
      response: JSON.stringify(parsed, null, 2)
    });
  }

  // ---------------------------------------------------
  // TEXT CHAT MODE
  // ---------------------------------------------------
  const body = await req.json();
  const messages: UIMessage[] = body.messages || [];

  const result = streamText({
    model: "gpt-4o-mini",
    messages: convertToModelMessages(messages),
    stopWhen: stepCountIs(10)
  });

  return result.toUIMessageStreamResponse();
}

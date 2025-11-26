// app/api/chat/route.ts
import OpenAI from "openai";
import { NextRequest } from "next/server";
import { UIMessage } from "ai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export const runtime = "edge"; // we don't use sharp to avoid node build issues

type OCRResponse = {
  extracted: string;
  analysis: string;
  response: string;
};

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ---------- IMAGE UPLOAD (multipart/form-data) ----------
  if (contentType.includes("multipart/form-data")) {
    try {
      const formData = await req.formData();
      const file = formData.get("image") as File | null;
      if (!file) {
        return new Response(JSON.stringify({ error: "No image uploaded." }), { status: 400 });
      }

      // Convert the browser File → base64 data URL (no server-side sharp)
      const ab = await file.arrayBuffer();
      const u8 = new Uint8Array(ab);
      // Node Buffer isn't necessary for edge runtime; use btoa on binary chunk
      let binary = "";
      const chunkSize = 0x8000;
      for (let i = 0; i < u8.length; i += chunkSize) {
        const slice = u8.subarray(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, Array.from(slice));
      }
      const base64 = typeof globalThis.btoa === "function"
        ? globalThis.btoa(binary)
        : Buffer.from(u8).toString("base64");
      const mime = file.type || "image/jpeg";
      const dataUrl = `data:${mime};base64,${base64}`;

      // ---------- 1) OCR via OpenAI Vision ----------
      // Ask model to *extract only* the ingredients text
      const ocrPrompt = `
Extract ONLY the ingredient list from the image. Return plain text only (comma or newline separated).
If you cannot find an "Ingredients" label, extract any text that looks like a list of ingredients.
Do NOT add commentary.
<image>${dataUrl}</image>
`;

      const ocrResp = await client.responses.create({
        model: "gpt-4.1-mini",
        input: ocrPrompt,
        temperature: 0
      });

      const extracted = (ocrResp.output_text || "").trim() || "UNREADABLE";

      // ---------- 2) FSSAI-style Analysis ----------
      const analysisPrompt = `
You are an Indian Food Safety analyst (FSSAI-aware).
Given the ingredient list below, classify each ingredient into categories:
- SAFE
- CAUTION
- POTENTIALLY_HARMFUL
- BANNED (India)
Also mark explicit allergens (e.g., MILK, SOY, WHEAT, NUTS, EGG) when present.
Return a concise plain-text analysis with:
1) Extracted ingredients (newline separated)
2) For each ingredient: classification and a one-line reason
3) A short "kid safety" note (yes/no + reason)
4) A summary safety score 0-10

Ingredient list:
${extracted}

Return plain text only (no JSON).
`;

      const analysisResp = await client.responses.create({
        model: "gpt-4.1-mini",
        input: analysisPrompt,
        temperature: 0
      });

      const analysis = (analysisResp.output_text || "").trim();

      // ---------- 3) Build final assistant text ----------
      const finalText =
        `📸 **Extracted Ingredients:**\n${extracted}\n\n` +
        `🔍 **FSSAI-style Safety Analysis:**\n${analysis}`;

      const result: OCRResponse = {
        extracted,
        analysis,
        response: finalText
      };

      return new Response(JSON.stringify(result), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    } catch (err: any) {
      console.error("Image handling error:", err?.message || err);
      return new Response(JSON.stringify({ error: "Server error processing image." }), { status: 500 });
    }
  }

  // ---------- NORMAL CHAT (JSON body) ----------
  try {
    const { messages } = await req.json();
    // If your app uses "ai" streaming tools, keep your existing chat logic here.
    // For simplicity we just echo or return a placeholder if you send messages.
    // You will likely keep your original streaming chat logic.
    return new Response(JSON.stringify({ echo: "chat mode not implemented in this example" }), { status: 200 });
  } catch (e) {
    return new Response(JSON.stringify({ error: "Bad request" }), { status: 400 });
  }
}

// app/api/chat/route.ts
import OpenAI from "openai";
import { NextResponse } from "next/server"; // Next.js App Router
import { Readable } from "stream";
import type { UIMessage } from "ai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export const runtime = "nodejs";
export const revalidate = 0;

async function fileToDataUrl(file: File) {
  const ab = await file.arrayBuffer();
  const u8 = new Uint8Array(ab);
  const base64 = Buffer.from(u8).toString("base64");
  return `data:${file.type};base64,${base64}`;
}

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // IMAGE UPLOAD flow
  if (contentType.includes("multipart/form-data")) {
    try {
      const formData = await req.formData();
      const file = formData.get("image") as File | null;

      if (!file) {
        return NextResponse.json({ error: "No image uploaded." }, { status: 400 });
      }

      // Convert to data URL (we do NOT use sharp here to avoid build/type issues)
      const dataUrl = await fileToDataUrl(file);

      // -------------------------
      // 1) OCR step - extract ingredients ONLY
      // -------------------------
      const ocrPrompt = `
You are an OCR assistant. EXTRACT ONLY the ingredient list from the image. 
Rules:
- Return ONLY the text after the word "Ingredients:" (or "INGREDIENTS", case-insensitive).
- If multiple lines, return them as a single newline-separated block.
- If no "Ingredients" header found, return any sequence of comma-separated or newline-separated ingredients you can read.
- Return plain text only (no commentary).
<image>${dataUrl}</image>
`.trim();

      const ocrResp = await client.responses.create({
        model: "gpt-4.1-mini",
        input: ocrPrompt,
        temperature: 0
      });

      const extracted =
        (ocrResp.output_text || "").trim().replace(/\r\n/g, "\n").replace(/\n{2,}/g, "\n");

      // -------------------------
      // 2) Analysis step - classify for FSSAI/allergen/child-safety
      // -------------------------
      const analysisPrompt = `
You are an Indian food-safety analyst. Parse the following ingredient text and return strict JSON only.

Input ingredient text:
"""${extracted}"""

Output JSON format EXACTLY like:
{
  "extracted": "...",             // string (the OCR result)
  "items": [
    { "name": "ingredient name", "fssai_status": "SAFE|HARMFUL|BANNED|UNKNOWN",
      "child_safe": true|false|null,
      "possible_allergens": ["milk","soy","wheat", ...],
      "notes": "short 1-line reason" }
  ],
  "summary": "short human summary",
  "score": 0-10 (number)
}

Rules & heuristics:
- If an ingredient is clearly an allergen (milk, egg, soy, wheat, peanut, tree nuts, sesame), list that in possible_allergens.
- If ingredient is a preservative/potent additive known in Indian lists (e.g., certain sulfites, borates, unauthorized additives) mark HARMFUL/BANNED if applicable.
- For ambiguous items, use UNKNOWN.
- child_safe true if commonly safe for children; false if known to be risky (alcohols, high % salicylates, high sodium nitrite etc).
- Keep output strictly JSON. No extra text.
`.trim();

      const analysisResp = await client.responses.create({
        model: "gpt-4.1-mini",
        input: analysisPrompt,
        temperature: 0,
        max_output_tokens: 800
      });

      const analysisText = analysisResp.output_text || "";
      let parsed;
      try {
        parsed = JSON.parse(analysisText);
      } catch (e) {
        // if model didn't return raw JSON, wrap fallback
        parsed = {
          extracted,
          items: [],
          summary: "Could not parse structured JSON from model.",
          raw: analysisText,
          score: 5
        };
      }

      return NextResponse.json({ ok: true, result: parsed });
    } catch (err: any) {
      return NextResponse.json({ ok: false, error: String(err?.message || err) }, { status: 500 });
    }
  }

  // TEXT CHAT flow (keeps existing behaviour)
  try {
    const body = await req.json();
    const messages = body.messages as UIMessage[] | undefined;

    // if you want, pass-through to GPT model for normal chat — minimal example:
    if (!messages || messages.length === 0) {
      return NextResponse.json({ ok: false, error: "No messages" }, { status: 400 });
    }

    // send to model (basic passthrough)
    const userText = messages.filter(m => m.role === "user").map(m =>
      (m.parts || []).map(p => (p as any).text || "").join("")).join("\n");

    const chatResp = await client.responses.create({
      model: "gpt-4.1-mini",
      input: userText
    });

    return NextResponse.json({ ok: true, response: chatResp.output_text || "" });
  } catch (err: any) {
    return NextResponse.json({ ok: false, error: String(err?.message || err) }, { status: 500 });
  }
}

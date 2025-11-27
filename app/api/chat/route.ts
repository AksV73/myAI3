import {
  streamText,
  convertToModelMessages,
  UIMessage,
  stepCountIs,
  createUIMessageStream,
  createUIMessageStreamResponse,
} from "ai";

import { MODEL } from "@/config";
import { SYSTEM_PROMPT } from "@/prompts";
import { isContentFlagged } from "@/lib/moderation";
import { webSearch } from "./tools/web-search";
import { vectorDatabaseSearch } from "./tools/search-vector-database";
import OpenAI from "openai";

export const maxDuration = 60;

// -------------------------
// Beautifier helper
// -------------------------
function beautifyMarkdown(text: string): string {
  if (!text) return text;
  return text
    .replace(/\. /g, ".\n\n") // add spacing after sentences
    .replace(/###/g, "\n###") // spacing before headings
    .replace(/\n{3,}/g, "\n\n") // normalize excessive breaks
    .trim();
}

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";

  // ======================================================
  // 📸 CASE 1 — IMAGE UPLOAD
  // ======================================================
  if (contentType.includes("multipart/form-data")) {
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
    const formData = await req.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return Response.json({ response: "No image found." });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const dataUrl = `data:${file.type};base64,${buffer.toString("base64")}`;

    // 1) Extract Ingredients using chat completions (image)
    const extractRes = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Extract ONLY the ingredient list from this image." },
            // Some OpenAI SDKs accept 'image_url' style; we keep as previously used
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
    });

    const extracted = extractRes.choices?.[0]?.message?.content || "No ingredients found.";

    // 2) Analyze Ingredients
    const analyzeRes = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `
You are an expert FSSAI Food Safety Consultant.

STYLE GUIDE:
- Use ## for headers.
- Use * for bullet points.
- Keep it concise.
- Start with a "Verdict" section.
`,
        },
        {
          role: "user",
          content: `Analyze these ingredients: ${extracted}`,
        },
      ],
    });

    const analysisRaw = analyzeRes.choices?.[0]?.message?.content || "Could not analyze.";
    const analysis = beautifyMarkdown(analysisRaw);

    return Response.json({
      response: `## 📸 Extracted Ingredients
${beautifyMarkdown(extracted)}

---

${analysis}`,
    });
  }

  // ======================================================
  // 💬 CASE 2 — TEXT CHAT (Streaming)
  // ======================================================
  const { messages }: { messages: UIMessage[] } = await req.json();

  const latest = messages.filter((m) => m.role === "user").pop();

  if (latest) {
    const textParts = latest.parts
      .filter((p) => p.type === "text")
      .map((p) => ("text" in p ? p.text : ""))
      .join("");

    const moderation = await isContentFlagged(textParts);

    if (moderation.flagged) {
      // Use the same safe streaming response pattern as your repo
      const stream = createUIMessageStream({
        execute({ writer }) {
          const id = "blocked";
          writer.write({ type: "start" });
          writer.write({ type: "text-start", id });
          writer.write({
            type: "text-delta",
            id,
            delta: moderation.denialMessage || "Message blocked.",
          });
          writer.write({ type: "text-end", id });
          writer.write({ type: "finish" });
        },
      });

      return createUIMessageStreamResponse({ stream });
    }
  }

  // ======================================================
  // 🤖 STREAMING CHAT RESPONSE
  // ======================================================
  const result = streamText({
    model: MODEL,
    system: `${SYSTEM_PROMPT}

STYLE GUIDE:
- Format all responses in clean Markdown.
- Use ## for main headers.
- Use * for bullet points.
- Keep sentences short.
- Avoid thick paragraphs.
`,
    messages: convertToModelMessages(messages),
    tools: { webSearch, vectorDatabaseSearch },
    // correct API (stopWhen) used to avoid maxSteps error
    stopWhen: stepCountIs(5),
  });

  // ----------------------------
  // Compatibility wrapper
  // ----------------------------
  // Many versions of the 'ai' SDK expose different streaming helpers.
  // We attempt the best transform path first (toDataStream -> experimental_transform),
  // and fall back to toUIMessageStreamResponse() when not available.
  //
  // We cast to `any` to avoid TypeScript compile errors for optional methods,
  // which keeps the file friendly across SDK versions.
  const anyResult = result as any;

  // 1) Preferred path: SDK exposes toDataStream() (allows chunk transforms)
  if (typeof anyResult.toDataStream === "function") {
    try {
      const stream = anyResult.toDataStream();

      // If experimental_transform exists, apply beautifier to text-delta chunks
      if (stream && typeof stream.experimental_transform === "function") {
        const transformed = stream.experimental_transform({
          transform(chunk: any) {
            try {
              if (chunk && chunk.type === "text-delta" && typeof chunk.text === "string") {
                chunk.text = beautifyMarkdown(chunk.text);
              }
            } catch (e) {
              // swallow transform errors so streaming still works
            }
            return chunk;
          },
        });

        // toResponse() exists on transformed stream in many SDK versions
        if (typeof transformed.toResponse === "function") {
          return transformed.toResponse();
        }

        // fallback: if transformed has toWebResponse or toResponse-like, try generic toResponse
        if (typeof (transformed as any).toWebResponse === "function") {
          return (transformed as any).toWebResponse();
        }
      }

      // If we couldn't transform, but stream has toResponse, return it (unchanged)
      if (typeof stream.toResponse === "function") {
        return stream.toResponse();
      }
    } catch (err) {
      // If anything here fails, we will fall through to the fallback path
      // (do not crash - keep response streaming working)
      console.warn("transform streaming path failed, falling back:", err);
    }
  }

  // 2) Fallback path: older SDKs expose toUIMessageStreamResponse()
  if (typeof anyResult.toUIMessageStreamResponse === "function") {
    // We cannot transform chunks here (API doesn't support transform).
    // Rely on SYSTEM prompt to format output as markdown.
    // Cast to any to avoid TS complaints about method signature differences.
    try {
      return anyResult.toUIMessageStreamResponse({ sendReasoning: true } as any);
    } catch (e) {
      // if signatures differ, attempt zero-arg call
      try {
        return anyResult.toUIMessageStreamResponse();
      } catch (e2) {
        // continue to final fallback
      }
    }
  }

  // 3) Final generic fallback: if result has toResponse, use it
  if (typeof anyResult.toResponse === "function") {
    return anyResult.toResponse();
  }

  // 4) As a last resort, if nothing else works, attempt to return a safety text stream
  // This ensures the endpoint doesn't crash the server; it returns a simple JSON response.
  return new Response(JSON.stringify({ error: "Streaming not supported by SDK at runtime." }), {
    status: 500,
    headers: { "content-type": "application/json" },
  });
}

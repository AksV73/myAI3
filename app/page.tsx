// app/page.tsx
"use client";

import React, { useRef, useState } from "react";
import { UIMessage } from "ai";

function resizeImageFileToBlob(file: File, maxWidth = 1600, quality = 0.85): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxWidth / img.width);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        return reject(new Error("Could not get canvas context"));
      }
      // White background useful for packaging labels with transparency
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(url);
        if (!blob) return reject(new Error("Failed to create blob"));
        resolve(blob);
      }, "image/jpeg", quality);
    };
    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(e);
    };
    img.src = url;
  });
}

export default function Page() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<UIMessage[]>([
    { id: `welcome-${Date.now()}`, role: "assistant", parts: [{ type: "text", text: "Hello — upload a label and I'll extract ingredients and run FSSAI checks." }] }
  ]);

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    setLoading(true);

    // Remove old OCR messages (IDs starting with ocr-)
    setMessages(prev => prev.filter(m => !String(m.id).startsWith("ocr-")));

    // Add temporary analyzing message:
    const analyzingId = `ocr-analyzing-${Date.now()}`;
    setMessages(prev => [...prev, { id: analyzingId, role: "assistant", parts: [{ type: "text", text: "📸 Analyzing image..." }] }]);

    try {
      // Client-side Resize (reduces upload size & avoids server sharp)
      let blob: Blob = f;
      try {
        blob = await resizeImageFileToBlob(f, 1600, 0.85);
      } catch {
        blob = f;
      }

      const fd = new FormData();
      // Use the original filename
      fd.append("image", blob, f.name);

      const res = await fetch("/api/chat", { method: "POST", body: fd });
      const json = await res.json();

      // remove analyzing message
      setMessages(prev => prev.filter(m => m.id !== analyzingId));

      if (!json.ok) {
        setMessages(prev => [...prev, { id: `ocr-error-${Date.now()}`, role: "assistant", parts: [{ type: "text", text: `Error: ${json.error || 'unknown'}` }] }]);
      } else {
        // Append structured result to messages (pretty printing)
        const pretty = typeof json.result === "string" ? json.result : JSON.stringify(json.result, null, 2);
        setMessages(prev => [...prev, { id: `ocr-${Date.now()}`, role: "assistant", parts: [{ type: "text", text: pretty }] }]);
      }
    } catch (err: any) {
      setMessages(prev => prev.filter(m => !String(m.id).startsWith("ocr-analyzing")));
      setMessages(prev => [...prev, { id: `ocr-error-${Date.now()}`, role: "assistant", parts: [{ type: "text", text: `Upload failed: ${String(err.message || err)}` }] }]);
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div style={{ padding: 32, maxWidth: 900, margin: "0 auto" }}>
      <h2>Ingredient OCR + FSSAI Check</h2>

      <div style={{ margin: "16px 0" }}>
        <input ref={fileRef} type="file" accept="image/*" onChange={onFileChange} />
        {loading && <div style={{ marginTop: 8 }}>Processing…</div>}
      </div>

      <div>
        {messages.map((m) => (
          <div key={String(m.id)} style={{ background: "#f3f4f6", padding: 12, borderRadius: 8, marginBottom: 10, whiteSpace: "pre-wrap" }}>
            <strong>{m.role === "assistant" ? "Bot" : m.role}</strong>
            <div style={{ marginTop: 6 }}>
              {m.parts?.map((p, i) => <div key={i}>{(p as any).text}</div>)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

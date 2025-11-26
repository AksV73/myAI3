// app/page.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { useForm, Controller } from "react-hook-form";
import { useChat } from "@ai-sdk/react";
import { UIMessage } from "ai";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FieldGroup, FieldLabel, Field } from "@/components/ui/field";
import { ArrowUp, Loader2, Square } from "lucide-react";

type FormValues = {
  message: string;
};

function resizeImageFile(file: File, maxWidth = 1400): Promise<Blob> {
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
        reject(new Error("No canvas context"));
        return;
      }
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (!blob) reject(new Error("Blob failed"));
          else resolve(blob);
        },
        "image/jpeg",
        0.87
      );
      URL.revokeObjectURL(url);
    };
    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(e);
    };
    img.src = url;
  });
}

export default function Page() {
  const form = useForm<FormValues>({ defaultValues: { message: "" } });
  const { messages, sendMessage, status, stop, setMessages } = useChat({
    messages: []
  });

  const fileRef = useRef<HTMLInputElement | null>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    // Optionally you can set a welcome message here
    if (messages.length === 0) {
      const welcome: UIMessage = {
        id: `welcome-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: "Hello! Upload an ingredient label or ask me anything." }]
      };
      setMessages((prev: UIMessage[]) => [...prev, welcome]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onSubmit(data: FormValues) {
    if (!data.message.trim()) return;
    sendMessage({ text: data.message });
    form.reset();
  }

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // 1) Resize client-side for speed
    let uploadBlob: Blob = file;
    try {
      uploadBlob = await resizeImageFile(file, 1400);
    } catch (err) {
      console.warn("Resize failed, using original file", err);
      uploadBlob = file;
    }

    const fd = new FormData();
    fd.append("image", uploadBlob, file.name);

    // Remove previous OCR messages we created (ids start with 'ocr-')
    setMessages((prev: UIMessage[]) => prev.filter((m) => !(String(m.id || "").startsWith("ocr-"))));

    // Add analyzing assistant message
    const analyzing: UIMessage = {
      id: `ocr-analyzing-${Date.now()}`,
      role: "assistant",
      parts: [{ type: "text", text: "📸 Analyzing image, extracting ingredients..." }]
    };
    setMessages((prev: UIMessage[]) => [...prev, analyzing]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        body: fd
      });
      const json = await res.json();

      // Remove analyzing message
      setMessages((prev: UIMessage[]) => prev.filter((m) => m.id !== analyzing.id));

      if (!res.ok) {
        const errMsg: UIMessage = {
          id: `ocr-error-${Date.now()}`,
          role: "assistant",
          parts: [{ type: "text", text: `Error: ${json?.error || res.statusText}` }]
        };
        setMessages((prev: UIMessage[]) => [...prev, errMsg]);
        return;
      }

      // Append assistant message with analysis
      const assistantMsg: UIMessage = {
        id: `ocr-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: json.response }]
      };
      setMessages((prev: UIMessage[]) => [...prev, assistantMsg]);
    } catch (err: any) {
      // Remove analyzing message
      setMessages((prev: UIMessage[]) => prev.filter((m) => m.id !== analyzing.id));
      const errMsg: UIMessage = {
        id: `ocr-error-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: `Upload failed: ${err.message || String(err)}` }]
      };
      setMessages((prev: UIMessage[]) => [...prev, errMsg]);
    } finally {
      // allow re-upload of same file
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="flex h-screen items-center justify-center font-sans bg-white">
      <main className="w-full max-w-3xl p-6">
        <div className="mb-4">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={onFileChange}
            className="border p-3 rounded w-full"
          />
        </div>

        <div className="space-y-3 mb-6">
          {messages.map((m) => (
            <div key={m.id} className={`p-4 rounded ${m.role === "assistant" ? "bg-gray-100" : "bg-blue-50"}`}>
              {m.parts?.map((p, idx) => p.type === "text" ? <div key={idx}>{p.text}</div> : null)}
            </div>
          ))}
        </div>

        <form onSubmit={form.handleSubmit(onSubmit)}>
          <FieldGroup>
            <Controller
              control={form.control}
              name="message"
              render={({ field }) => (
                <Field>
                  <FieldLabel className="sr-only">Message</FieldLabel>
                  <div className="relative">
                    <Input {...field} placeholder="Type your message..." className="pr-12" />
                    <div className="absolute right-2 top-1">
                      {(status === "ready" || status === "error") && (
                        <Button type="submit" size="icon" disabled={!field.value.trim()}>
                          <ArrowUp />
                        </Button>
                      )}
                      {(status === "streaming" || status === "submitted") && (
                        <Button size="icon" onClick={() => stop()}>
                          <Square />
                        </Button>
                      )}
                    </div>
                  </div>
                </Field>
              )}
            />
          </FieldGroup>
        </form>
      </main>
    </div>
  );
}

// app/page.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { useForm, Controller } from "react-hook-form";
import { useChat } from "@ai-sdk/react";
import { UIMessage } from "ai";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FieldGroup, FieldLabel, Field } from "@/components/ui/field";
import { ArrowUp, Square } from "lucide-react";

type FormValues = { message: string };

// ------------------------
// OPTIONAL CLIENT RESIZE
// ------------------------
function resizeImageFile(file: File, maxWidth = 1400): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      const scale = Math.min(1, maxWidth / img.width);
      const canvas = document.createElement("canvas");
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      const ctx = canvas.getContext("2d");
      if (!ctx) return reject("No canvas");

      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject("Blob failed")),
        "image/jpeg",
        0.9
      );

      URL.revokeObjectURL(url);
    };

    img.onerror = reject;
    img.src = url;
  });
}

export default function Page() {
  const form = useForm<FormValues>({ defaultValues: { message: "" } });

  // ❗ FIXED: no "messages: []" here
  const { messages, sendMessage, status, stop, setMessages } = useChat();

  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (messages.length === 0) {
      const welcome: UIMessage = {
        id: `welcome-${Date.now()}`,
        role: "assistant",
        parts: [
          {
            type: "text",
            text: "👋 Hi! Upload a food label image or ask me anything."
          }
        ]
      };
      setMessages([welcome]);
    }
  }, []);

  // ------------------------
  // SEND TEXT MESSAGE
  // ------------------------
  async function onSubmit(data: FormValues) {
    if (!data.message.trim()) return;
    sendMessage({ text: data.message });
    form.reset();
  }

  // ------------------------
  // IMAGE UPLOAD HANDLER
  // ------------------------
  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Resize client-side
    let processed: Blob = file;
    try {
      processed = await resizeImageFile(file, 1400);
    } catch (_) {}

    const fd = new FormData();
    fd.append("image", processed, file.name);

    // Add analyzing message
    const thinking: UIMessage = {
      id: `ocr-analyzing-${Date.now()}`,
      role: "assistant",
      parts: [{ type: "text", text: "📸 Reading label... please wait." }]
    };
    setMessages((prev) => [...prev, thinking]);

    try {
      const res = await fetch("/api/chat", { method: "POST", body: fd });
      const json = await res.json();

      // Remove analyzing message
      setMessages((prev) => prev.filter((m) => m.id !== thinking.id));

      const out: UIMessage = {
        id: `ocr-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: json.response }]
      };
      setMessages((prev) => [...prev, out]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `ocr-error-${Date.now()}`,
          role: "assistant",
          parts: [{ type: "text", text: `❌ Upload failed: ${String(err)}` }]
        }
      ]);
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-white">
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
            <div
              key={m.id}
              className={`p-4 rounded ${
                m.role === "assistant" ? "bg-gray-100" : "bg-blue-100"
              }`}
            >
              {m.parts?.map((p, i) =>
                p.type === "text" ? <div key={i}>{p.text}</div> : null
              )}
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
                    <Input {...field} placeholder="Type your message..." />

                    <div className="absolute right-2 top-1">
                      {status === "ready" && (
                        <Button type="submit" size="icon">
                          <ArrowUp />
                        </Button>
                      )}
                      {(status === "submitted" ||
                        status === "streaming") && (
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

"use client";

import { Controller, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useEffect, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { UIMessage } from "ai";
import { Button } from "@/components/ui/button";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { ArrowUp, Loader2, Plus, Square } from "lucide-react";
import Image from "next/image";
import { AI_NAME, CLEAR_CHAT_TEXT, OWNER_NAME, WELCOME_MESSAGE } from "@/config";

const formSchema = z.object({
  message: z.string().min(1).max(2000),
});

function resizeImageFile(file: File, maxWidth = 1600): Promise<Blob> {
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
        reject(new Error("No canvas"));
        return;
      }
      // white background for transparent labels
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (!blob) reject(new Error("Blob failed"));
          else resolve(blob);
        },
        "image/jpeg",
        0.9
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

export default function ChatPage() {
  const form = useForm({ resolver: zodResolver(formSchema), defaultValues: { message: "" } });
  const { messages, sendMessage, status, stop, setMessages } = useChat({ messages: [] as UIMessage[] });
  const [isClient, setIsClient] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setIsClient(true);
    // optional: set welcome message once
    if (messages.length === 0) {
      const welcome: UIMessage = {
        id: `welcome-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: WELCOME_MESSAGE }],
      };
      setMessages([welcome]);
    }
  }, []);

  async function onSubmit(data: any) {
    sendMessage({ text: data.message });
    form.reset();
  }

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Resize client-side for speed + reliability
    let uploadBlob: Blob = file;
    try {
      uploadBlob = await resizeImageFile(file, 1600);
    } catch (err) {
      console.warn("Client resize failed, using original", err);
      uploadBlob = file;
    }

    const fd = new FormData();
    fd.append("image", uploadBlob, file.name);

    // Remove old OCR messages (custom ids starting with ocr-)
    setMessages((prev) => prev.filter((m) => !String(m.id).startsWith("ocr-")));

    // temporary analyzing message
    const analyzing: UIMessage = {
      id: `ocr-analyzing-${Date.now()}`,
      role: "assistant",
      parts: [{ type: "text", text: "📸 Analyzing ingredients..." }],
    };
    setMessages((prev) => [...prev, analyzing]);

    try {
      const res = await fetch("/api/chat", { method: "POST", body: fd });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Status ${res.status}`);
      }
      const json = await res.json();

      // remove analyzing message
      setMessages((prev) => prev.filter((m) => m.id !== analyzing.id));

      // Append final result message (structured)
      const finalMsg: UIMessage = {
        id: `ocr-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: json.response }],
      };
      setMessages((prev) => [...prev, finalMsg]);
    } catch (err: any) {
      // remove analyzing message
      setMessages((prev) => prev.filter((m) => m.id !== analyzing.id));
      const errMsg: UIMessage = {
        id: `ocr-error-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: `Error processing image: ${err.message || String(err)}` }],
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      // allow uploading same file again
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <div className="flex h-screen items-center justify-center">
      <main className="w-full max-w-4xl px-4">
        <div className="py-6">
          {/* header and message wall omitted for brevity - use your existing ChatHeader & MessageWall */}
          {/* File input */}
          <div className="mb-3">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFile}
              className="border p-2 rounded"
            />
          </div>

          {/* Text chat form */}
          <form id="chat-form" onSubmit={form.handleSubmit(onSubmit)}>
            <FieldGroup>
              <Controller
                control={form.control}
                name="message"
                render={({ field }) => (
                  <div className="relative">
                    <Input {...field} placeholder="Type your message..." />
                    <div className="absolute right-2 top-1">
                      <Button type="submit" size="icon" disabled={!field.value.trim()}>
                        <ArrowUp />
                      </Button>
                    </div>
                  </div>
                )}
              />
            </FieldGroup>
          </form>
        </div>
      </main>
    </div>
  );
}

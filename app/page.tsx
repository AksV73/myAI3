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
import { ArrowUp, Loader2, Square } from "lucide-react";

import {
  AI_NAME,
  CLEAR_CHAT_TEXT,
  OWNER_NAME,
  WELCOME_MESSAGE,
} from "@/config";

const formSchema = z.object({
  message: z.string().min(1).max(2000),
});

// ---------------------------------------------
// CLIENT-SIDE IMAGE RESIZER (TYPE-SAFE)
// ---------------------------------------------
function resizeImageFile(file: File, maxWidth = 1600): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = document.createElement("img"); // <-- FIXED TYPE

    img.onload = () => {
      const scale = Math.min(1, maxWidth / img.width);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas error"));
        return;
      }

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
  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: { message: "" },
  });

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const { messages, sendMessage, status, stop, setMessages } = useChat({
    messages: [] as UIMessage[],
  });

  const [isClient, setIsClient] = useState(false);

  // ---------------------------------------------
  // INITIAL WELCOME MESSAGE
  // ---------------------------------------------
  useEffect(() => {
    setIsClient(true);

    if (messages.length === 0) {
      const welcome: UIMessage = {
        id: `welcome-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: WELCOME_MESSAGE }],
      };
      setMessages([welcome]);
    }
  }, []);

  // ---------------------------------------------
  // TEXT MESSAGE SUBMIT
  // ---------------------------------------------
  function onSubmit(data: any) {
    sendMessage({ text: data.message });
    form.reset();
  }

  // ---------------------------------------------
  // IMAGE UPLOAD HANDLER
  // ---------------------------------------------
  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    let resizedBlob: Blob = file;

    try {
      resizedBlob = await resizeImageFile(file, 1600);
    } catch (err) {
      console.warn("Resize failed. Using original image.");
      resizedBlob = file;
    }

    const fd = new FormData();
    fd.append("image", resizedBlob, file.name);

    // remove previous OCR results
    setMessages((prev) =>
      prev.filter((m) => !String(m.id).startsWith("ocr-"))
    );

    const analyzing: UIMessage = {
      id: `ocr-analyzing-${Date.now()}`,
      role: "assistant",
      parts: [{ type: "text", text: "📸 Analyzing image..." }],
    };

    setMessages((prev) => [...prev, analyzing]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error(await res.text());

      const json = await res.json();

      setMessages((prev) => prev.filter((m) => m.id !== analyzing.id));

      const result: UIMessage = {
        id: `ocr-${Date.now()}`,
        role: "assistant",
        parts: [{ type: "text", text: json.response }],
      };

      setMessages((prev) => [...prev, result]);
    } catch (err: any) {
      setMessages((prev) => prev.filter((m) => m.id !== analyzing.id));

      const errorMsg: UIMessage = {
        id: `ocr-error-${Date.now()}`,
        role: "assistant",
        parts: [
          { type: "text", text: `❌ Error analyzing image: ${err.message}` },
        ],
      };

      setMessages((prev) => [...prev, errorMsg]);
    }

    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // ---------------------------------------------
  // UI
  // ---------------------------------------------
  return (
    <div className="flex h-screen items-center justify-center">
      <main className="w-full max-w-4xl px-4 py-6">

        {/* IMAGE UPLOAD */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFile}
          className="border p-2 rounded mb-5"
        />

        {/* CHAT MESSAGES */}
        <div className="space-y-3 mb-20">
          {messages.map((msg) => (
            <div key={msg.id} className="p-3 bg-gray-100 rounded">
              {msg.parts.map((p, i) => (
                <p key={i}>{p.type === "text" ? p.text : ""}</p>
              ))}
            </div>
          ))}

          {status === "streaming" && (
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          )}
        </div>

        {/* TEXT INPUT */}
        <form onSubmit={form.handleSubmit(onSubmit)}>
          <FieldGroup>
            <Controller
              control={form.control}
              name="message"
              render={({ field }) => (
                <div className="relative">
                  <Input {...field} placeholder="Type your message..." />
                  <Button
                    type="submit"
                    size="icon"
                    disabled={!field.value.trim()}
                    className="absolute right-2 top-1"
                  >
                    <ArrowUp />
                  </Button>
                </div>
              )}
            />
          </FieldGroup>
        </form>
      </main>
    </div>
  );
}

"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Controller, useForm } from "react-hook-form";
import { toast } from "sonner";
import * as z from "zod";

import { Button } from "@/components/ui/button";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { useChat } from "@ai-sdk/react";

import { ArrowUp, Loader2, Plus, Square } from "lucide-react";
import { MessageWall } from "@/components/messages/message-wall";
import { ChatHeader, ChatHeaderBlock } from "@/app/parts/chat-header";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

import { UIMessage } from "ai";
import { useEffect, useState, useRef } from "react";
import { AI_NAME, CLEAR_CHAT_TEXT, OWNER_NAME, WELCOME_MESSAGE } from "@/config";

import Image from "next/image";
import Link from "next/link";


// -------------------------------
// FORM SCHEMA
// -------------------------------
const formSchema = z.object({
  message: z.string().min(1).max(2000),
});

// -------------------------------
// LOCAL STORAGE HELPERS
// -------------------------------
const STORAGE_KEY = "chat-messages";

type StorageData = {
  messages: UIMessage[];
  durations: Record<string, number>;
};

const loadMessages = (): StorageData => {
  if (typeof window === "undefined") return { messages: [], durations: {} };
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") || {
      messages: [],
      durations: {},
    };
  } catch {
    return { messages: [], durations: {} };
  }
};

const saveMessages = (messages: UIMessage[], durations: Record<string, number>) => {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ messages, durations }));
};


// -------------------------------
// COMPONENT
// -------------------------------
export default function Chat() {
  const [isClient, setIsClient] = useState(false);
  const welcomeRef = useRef(false);

  const stored = typeof window !== "undefined" ? loadMessages() : { messages: [], durations: {} };

  const [durations, setDurations] = useState<Record<string, number>>(stored.durations);
  const [initialMessages] = useState<UIMessage[]>(stored.messages);

  const { messages, sendMessage, status, stop, setMessages } = useChat({
    messages: initialMessages,
  });

  useEffect(() => {
    setIsClient(true);
    setDurations(stored.durations);
    setMessages(stored.messages);
  }, []);

  useEffect(() => {
    if (isClient) saveMessages(messages, durations);
  }, [messages, durations, isClient]);

  // WELCOME MESSAGE
  useEffect(() => {
    if (!isClient || initialMessages.length !== 0 || welcomeRef.current) return;

    const welcomeMessage: UIMessage = {
      id: "welcome-" + Date.now(),
      role: "assistant",
      parts: [{ type: "text", text: WELCOME_MESSAGE }],
    };

    setMessages([welcomeMessage]);
    saveMessages([welcomeMessage], {});
    welcomeRef.current = true;
  }, [isClient, initialMessages.length, setMessages]);


  // FORM
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { message: "" },
  });

  function onSubmit(data: z.infer<typeof formSchema>) {
    sendMessage({ text: data.message });
    form.reset();
  }

  function clearChat() {
    setMessages([]);
    setDurations({});
    saveMessages([], {});
    toast.success("Chat cleared");
  }


  return (
    <div className="flex h-screen items-center justify-center font-sans dark:bg-black">
      <main className="w-full dark:bg-black h-screen relative">

        {/* HEADER */}
        <div className="fixed top-0 left-0 right-0 z-50 pb-16 bg-linear-to-b from-background via-background/50 to-transparent dark:bg-black">
          <ChatHeader>
            <ChatHeaderBlock />
            <ChatHeaderBlock className="justify-center items-center">
              <Avatar className="size-8 ring-1 ring-primary">
                <AvatarImage src="/logo.png" />
                <AvatarFallback>
                  <Image src="/logo.png" width={36} height={36} alt="logo" />
                </AvatarFallback>
              </Avatar>
              <p className="tracking-tight">Chat with {AI_NAME}</p>
            </ChatHeaderBlock>

            <ChatHeaderBlock className="justify-end">
              <Button variant="outline" size="sm" onClick={clearChat}>
                <Plus className="size-4" />
                {CLEAR_CHAT_TEXT}
              </Button>
            </ChatHeaderBlock>
          </ChatHeader>
        </div>

        {/* MESSAGES */}
        <div className="h-screen overflow-y-auto px-5 py-4 w-full pt-[88px] pb-[150px]">
          <div className="flex flex-col items-center justify-end min-h-full">
            {isClient ? (
              <>
                <MessageWall
                  messages={messages}
                  status={status}
                  durations={durations}
                  onDurationChange={(key, d) =>
                    setDurations((prev: Record<string, number>) => ({
                      ...prev,
                      [key]: d,
                    }))
                  }
                />

                {status === "submitted" && (
                  <div className="flex justify-start max-w-3xl w-full">
                    <Loader2 className="size-4 animate-spin text-muted-foreground" />
                  </div>
                )}
              </>
            ) : (
              <Loader2 className="animate-spin size-4 text-muted-foreground" />
            )}
          </div>
        </div>

        {/* INPUT AREA */}
        <div className="fixed bottom-0 left-0 right-0 z-50 pt-13 bg-linear-to-t from-background via-background/50 to-transparent dark:bg-black">

          <div className="w-full px-5 pt-5 pb-1 flex justify-center">
            <div className="max-w-3xl w-full">

              <form onSubmit={form.handleSubmit(onSubmit)}>

                {/* IMAGE UPLOAD */}
                <input
                  type="file"
                  accept="image/*"
                  capture="environment"
                  className="border p-2 rounded mb-3"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;

                    const formData = new FormData();
                    formData.append("image", file);

                    // temporary message
                    setMessages((prev) => [
                      ...prev,
                      {
                        id: "temp-" + Date.now(),
                        role: "assistant",
                        parts: [{ type: "text", text: "📸 Analyzing image…" }],
                      },
                    ]);

                    const res = await fetch("/api/chat", {
                      method: "POST",
                      body: formData,
                    });

                    const data = await res.json();

                    // final output message
                    setMessages((prev) => [
                      ...prev,
                      {
                        id: "ans-" + Date.now(),
                        role: "assistant",
                        parts: [{ type: "text", text: data.response }],
                      },
                    ]);
                  }}
                />

                {/* TEXT INPUT */}
                <FieldGroup>
                  <Controller
                    name="message"
                    control={form.control}
                    render={({ field, fieldState }) => (
                      <Field data-invalid={fieldState.invalid}>
                        <FieldLabel className="sr-only">Message</FieldLabel>

                        <div className="relative h-13">
                          <Input
                            {...field}
                            placeholder="Type your message..."
                            className="h-15 pr-15 pl-5 bg-card rounded-[20px]"
                            disabled={status === "streaming"}
                            autoComplete="off"
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                form.handleSubmit(onSubmit)();
                              }
                            }}
                          />

                          {(status === "ready" || status === "error") && (
                            <Button
                              type="submit"
                              className="absolute right-3 top-3 rounded-full"
                              size="icon"
                              disabled={!field.value.trim()}
                            >
                              <ArrowUp className="size-4" />
                            </Button>
                          )}

                          {(status === "streaming" || status === "submitted") && (
                            <Button
                              className="absolute right-2 top-2 rounded-full"
                              size="icon"
                              onClick={stop}
                            >
                              <Square className="size-4" />
                            </Button>
                          )}
                        </div>
                      </Field>
                    )}
                  />
                </FieldGroup>

              </form>
            </div>
          </div>

          {/* FOOTER */}
          <div className="w-full px-5 py-3 text-xs text-muted-foreground flex justify-center">
            © {new Date().getFullYear()} {OWNER_NAME} ·
            <Link href="/terms" className="underline">&nbsp;Terms</Link> ·
            Powered by&nbsp;<Link href="https://ringel.ai/" className="underline">Ringel.AI</Link>
          </div>

        </div>
      </main>
    </div>
  );
}

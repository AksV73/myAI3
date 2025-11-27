"use client";

interface AssistantMessageProps {
  message: {
    id: string;
    role: string;
    content: string;
  };
}

/**
 * Lightweight Markdown → HTML converter
 * NO imports, NO external packages, NO terminal required.
 */
function markdownToHtml(md: string): string {
  if (!md) return "";

  let html = md;

  // --- Headings ---
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

  // --- Bold + italic ---
  html = html.replace(/\*\*\*(.*?)\*\*\*/gim, "<strong><em>$1</em></strong>");

  // --- Bold ---
  html = html.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");

  // --- Italic ---
  html = html.replace(/\*(.*?)\*/gim, "<em>$1</em>");

  // --- Bullet points ---
  html = html.replace(/^\* (.*$)/gim, "<li>$1</li>");

  // Wrap <li> inside <ul> group
  html = html.replace(/(<li>[\s\S]*?<\/li>)/gim, "<ul>$1</ul>");

  // --- Line breaks ---
  html = html.replace(/\n/g, "<br>");

  return html.trim();
}

export default function AssistantMessage({ message }: AssistantMessageProps) {
  return (
    <div
      className="p-4 bg-gray-100 rounded-lg text-sm leading-relaxed whitespace-pre-wrap"
      dangerouslySetInnerHTML={{ __html: markdownToHtml(message.content) }}
    />
  );
}

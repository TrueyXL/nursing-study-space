// Nursing Study Space AI proxy.
// Holds the Anthropic API key (secret ANTHROPIC_API_KEY) so the public GitHub site never sees it.
// Callers must send the class code (secret CLASS_CODE) and come from an allowed origin.
// Replies stream back as plain text; an error after streaming starts is sent as "\u001e" + JSON.
import Anthropic from "@anthropic-ai/sdk";

interface Env {
  ANTHROPIC_API_KEY: string;
  CLASS_CODE: string;
  ALLOWED_ORIGINS: string;
  MODEL?: string;
  LIMITER?: { limit(opts: { key: string }): Promise<{ success: boolean }> };
}

const SYSTEM =
  "You are the study assistant inside a nursing students' study app (Canadian nursing program). " +
  "Help with nursing, health sciences, clinical reasoning, exam prep, study skills, academic writing for nursing courses, clinical French, and student wellbeing. " +
  "Politely decline requests unrelated to those topics. This is for studying, not real patient care.";

const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);
const ERR = "\u001e";

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const origin = req.headers.get("Origin") || "";
    const allowed = (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
    const cors: Record<string, string> = {
      "Access-Control-Allow-Origin": allowed.includes(origin) ? origin : allowed[0] || "",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Vary": "Origin",
    };
    const json = (status: number, body: unknown) =>
      new Response(JSON.stringify(body), { status, headers: { ...cors, "Content-Type": "application/json" } });

    if (req.method === "OPTIONS") return new Response(null, { headers: cors });
    if (req.method !== "POST") return json(405, { error: "method_not_allowed" });
    if (!allowed.includes(origin)) return json(403, { error: "origin_not_allowed" });
    if (Number(req.headers.get("Content-Length") || 0) > 15_000_000) return json(413, { error: "too_large" });

    let body: any;
    try { body = await req.json(); } catch { return json(400, { error: "bad_json" }); }
    if (!env.CLASS_CODE || typeof body?.code !== "string" || body.code.trim().toLowerCase() !== env.CLASS_CODE.trim().toLowerCase()) {
      return json(401, { error: "bad_code" });
    }
    if (env.LIMITER) {
      const { success } = await env.LIMITER.limit({ key: req.headers.get("CF-Connecting-IP") || "unknown" });
      if (!success) return json(429, { error: "rate_limited" });
    }

    // Keep only well-formed text turns, newest 30, each capped in length.
    const messages: Anthropic.Beta.BetaMessageParam[] = (Array.isArray(body.messages) ? body.messages : [])
      .filter((m: any) => (m?.role === "user" || m?.role === "assistant") && typeof m.content === "string" && m.content.trim())
      .slice(-30)
      .map((m: any) => ({ role: m.role, content: m.content.slice(0, 40_000) }));
    while (messages.length && messages[0].role !== "user") messages.shift();
    if (!messages.length || messages[messages.length - 1].role !== "user") return json(400, { error: "bad_messages" });

    const images = (Array.isArray(body.images) ? body.images : [])
      .filter((i: any) => IMAGE_TYPES.has(i?.media_type) && typeof i.data === "string")
      .slice(0, 4);
    if (images.length) {
      const last = messages[messages.length - 1];
      last.content = [
        ...images.map((i: any) => ({ type: "image" as const, source: { type: "base64" as const, media_type: i.media_type, data: i.data } })),
        { type: "text" as const, text: last.content as string },
      ];
    }

    if (!env.ANTHROPIC_API_KEY) return json(503, { error: "server_key" });
    const quick = body.mode === "json";
    const model = env.MODEL || "claude-opus-5-5";
    // Haiku 5.5 has no server-side refusal fallback; the other current models do.
    const fallback = !model.includes("haiku");
    const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
    const params: any = {
      model,
      max_tokens: quick ? 8000 : 16000,
      system: SYSTEM,
      output_config: { effort: quick ? "low" : "medium" },
      messages,
    };
    const stream = fallback
      ? client.beta.messages.stream({ ...params, betas: ["server-side-fallback-2026-07-01"], fallbacks: "default" })
      : client.beta.messages.stream(params);

    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const enc = new TextEncoder();
    ctx.waitUntil((async () => {
      try {
        stream.on("text", (t: string) => { writer.write(enc.encode(t)); });
        const final = await stream.finalMessage();
        if (final.stop_reason === "refusal") await writer.write(enc.encode(ERR + JSON.stringify({ error: "refusal" })));
        else if (final.stop_reason === "max_tokens") await writer.write(enc.encode("\n\n_(The answer was cut off. Ask me to continue.)_"));
      } catch (e) {
        const code = e instanceof Anthropic.RateLimitError ? "busy"
          : e instanceof Anthropic.AuthenticationError ? "server_key"
          : e instanceof Anthropic.BadRequestError ? "bad_request"
          : e instanceof Anthropic.APIError && (e.status ?? 0) >= 500 ? "busy"
          : "error";
        console.error("anthropic error", code, (e as Error)?.message);
        await writer.write(enc.encode(ERR + JSON.stringify({ error: code })));
      } finally {
        await writer.close();
      }
    })());
    return new Response(readable, { headers: { ...cors, "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" } });
  },
};

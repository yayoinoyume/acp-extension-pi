import { existsSync } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import * as acp from "@agentclientprotocol/sdk";
import { z } from "zod";

import {
  LODY_EXTENSION_METHODS,
  SessionUsageAccumulator,
  LodySubagentEmitter,
  LODY_SUBAGENT_EVENT_METHOD,
  type LodyActivityMeta,
  type SessionUsageUpdate,
} from "acp-extension-core";
import type { AgentConnection, PiStream } from "./types.js";
import { PiTransport } from "./transport.js";
import {
  piAssistantUsageSchema,
  piModelKey,
  piResponseOperation,
  piUsageSchema,
  toModelUsage,
} from "./usage.js";
import { PI_RPC_VERSION } from "./version.js";
import {
  detectCapabilities,
  liveProbes,
  type PiCapabilities,
} from "./capability.js";
import { QUESTION_PREFIX, questionsSchema } from "./builtin-tools.js";

const modelSchema = z.object({
  provider: z.string(),
  id: z.string(),
  name: z.string(),
});
const stateSchema = z.object({
  sessionFile: z.string().optional(),
  model: modelSchema.nullish(),
  thinkingLevel: z.string(),
  isStreaming: z.boolean().optional(),
  isCompacting: z.boolean().optional(),
  pendingMessageCount: z.number().optional(),
});
const contentSchema = z.array(
  z.discriminatedUnion("type", [
    z.object({ type: z.literal("text"), text: z.string() }),
    z.object({
      type: z.literal("image"),
      data: z.string(),
      mimeType: z.string(),
    }),
  ]),
);
const statsSchema = z.object({
  tokens: z.object({
    input: z.number().nonnegative(),
    output: z.number().nonnegative(),
    cacheRead: z.number().nonnegative(),
    cacheWrite: z.number().nonnegative(),
  }),
  cost: z.number().nonnegative(),
  contextUsage: z
    .object({
      tokens: z.number().nonnegative().nullable(),
      contextWindow: z.number().nonnegative(),
    })
    .optional(),
});
const commandsSchema = z.object({
  commands: z.array(
    z.object({
      name: z.string().min(1),
      description: z.string().nullish(),
      source: z.string().nullish(),
    }),
  ),
});
/** Marks the dispatch rejection itself so RPC failures stay non-fatal. */
class ExtensionCommandError extends Error {
  constructor(name: string) {
    super(
      `/${name} is a Pi extension command; Lody cannot dispatch it safely. Run it in a Pi terminal instead.`,
    );
    this.name = "ExtensionCommandError";
  }
}

/** Lody reads the command list from the session response, a deliberate
 * extension of the standard ACP payload that the SDK types predate. */
type WithCommands<T> = T & {
  availableCommands: acp.AvailableCommand[];
};
type SessionResponseWithCommands = WithCommands<acp.NewSessionResponse>;
/** How a handled prompt waits for a detached run: poll interval, how long the
 * runtime must stay quiet, and the ceiling no single extension may exceed. */
const DETACHED_POLL_MS = 50;
const DETACHED_QUIET_MS = 3000;
const DETACHED_MAX_WINDOW_MS = 5000;
const LOCAL_COMMANDS = [
  { name: "compact", description: "Manually compact the session context" },
  { name: "stats", description: "Show Pi session usage and cost" },
];
const startupConfigSchema = z.object({
  lody: z.object({
    sessionConfig: z.object({
      configOptionValues: z.record(
        z.string(),
        z.union([z.string(), z.boolean()]),
      ),
    }),
  }),
});
const questionSchema = z.object({
  id: z.string(),
  method: z.string(),
  title: z.string().optional(),
});
const todosSchema = z.object({
  todos: z.array(z.object({ text: z.string(), done: z.boolean() })),
});
const userContentSchema = z.union([z.string(), contentSchema]);
const entriesSchema = z.object({
  entries: z.array(
    z.object({ id: z.string(), parentId: z.string().nullable() }).passthrough(),
  ),
  leafId: z.string().nullable(),
});
const userEntrySchema = z.union([
  z.object({
    type: z.literal("message"),
    message: z.object({ role: z.literal("user"), content: userContentSchema }),
  }),
  z.object({
    type: z.literal("custom_message"),
    customType: z.literal("lody-steer"),
    content: userContentSchema,
  }),
]);
const assistantEntrySchema = z.object({
  type: z.literal("message"),
  message: z.object({
    role: z.literal("assistant"),
    content: z.array(z.unknown()),
  }),
});
const assistantBlockSchema = z.union([
  z.object({ type: z.literal("text"), text: z.string() }),
  z.object({ type: z.literal("thinking"), thinking: z.string() }),
  z.object({
    type: z.literal("toolCall"),
    id: z.string(),
    name: z.string(),
    arguments: z.record(z.string(), z.unknown()),
  }),
]);
const toolResultEntrySchema = z.object({
  type: z.literal("message"),
  message: z.object({
    role: z.literal("toolResult"),
    toolCallId: z.string(),
    content: contentSchema,
    details: z.unknown().optional(),
    isError: z.boolean().optional(),
  }),
});

function planEntries({ todos }: z.infer<typeof todosSchema>): acp.PlanEntry[] {
  return todos.map((todo) => ({
    content: todo.text,
    priority: "medium",
    status: todo.done ? "completed" : "pending",
  }));
}

function userContent(
  content: z.infer<typeof userContentSchema>,
): acp.ContentBlock[] {
  if (typeof content === "string")
    return content ? [{ type: "text", text: content }] : [];
  const text = content
    .flatMap((block) => (block.type === "text" ? block.text : []))
    .join("\n\n");
  return [
    ...(text ? [{ type: "text" as const, text }] : []),
    ...content.filter((block) => block.type === "image"),
  ];
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  // A process may exit while the caller is still awaiting the command acknowledgement.
  void promise.catch(() => undefined);
  return { promise, resolve, reject };
}
type Progress = {
  activities: Map<string, { id: string; meta: LodyActivityMeta }>;
  started: boolean;
  modelSettled: boolean;
  completed: boolean;
  cancelled: boolean;
  error?: string;
  stopReason?: "max_tokens";
};
type Run = Progress &
  ReturnType<typeof deferred<acp.PromptResponse>> & {
    finished: ReturnType<typeof deferred<void>>;
    cancellation?: Promise<void>;
    settled: ReturnType<typeof deferred<void>>;
    accepted?: Promise<unknown>;
    /** The prompt text, echoed for notices that name the command. */
    message: string;
    /** What the runtime reported about dispatch, when it reports at all. */
    disposition?: string;
  };
type Host = {
  subagentEvents?: boolean;
  configureMcp: (servers: acp.McpServer[]) => Promise<void>;
  update: (notification: acp.SessionNotification) => Promise<void>;
  extension: (method: string, params: Record<string, unknown>) => Promise<void>;
  usage: (usage: SessionUsageUpdate) => void;
  question: (
    request: acp.CreateElicitationRequest,
  ) => Promise<acp.CreateElicitationResponse>;
};

/** Translates the ACP control contract to the pinned Pi JSONL runtime. */
export class PiRpcConnection implements AgentConnection {
  private readonly subagentRuns = new Map<string, LodySubagentEmitter>();
  private subagents(): LodySubagentEmitter {
    let runs = this.subagentRuns.get(this.sessionId);
    if (!runs) {
      runs = new LodySubagentEmitter(this.sessionId, (event) =>
        this.host.extension(LODY_SUBAGENT_EVENT_METHOD, { ...event }),
      );
      this.subagentRuns.set(this.sessionId, runs);
    }
    return runs;
  }
  private readonly rpc: PiTransport;
  private sessionId = "";
  private cwd = "";
  private active?: Run;
  private configuring = false;
  private readonly questions = new Map<string, () => Promise<void>>();
  private readonly controls = new Map<
    string,
    ReturnType<typeof deferred<unknown>>
  >();
  private pendingPlan?: acp.PlanEntry[];
  /** Live per-response usage of this activation. Pi stats are a sum over the
   * current context, not a counter: compaction, retries and branching lower them
   * and resume restores history, so they never feed accounting. */
  private usageLedger = new SessionUsageAccumulator();
  private compactions = 0;
  private lastModelKey?: string;
  private steerCommand = "";
  private capabilities: PiCapabilities = { hasDisposition: false };
  private capabilityProbe?: Promise<PiCapabilities>;
  /** No disposition means either a runtime older than 0.99.0 or a failed
   * probe; both keep the safe path that withholds extension commands. */
  private get legacy(): boolean {
    return !this.capabilities.hasDisposition;
  }
  /** A turn closed while the runtime was idle. Its late events are orphans. */
  private closedIdleRun?: Run;
  private pendingSteer?: {
    id: string;
    run: Run;
    applied: ReturnType<typeof deferred<void>>;
  };

  constructor(
    stream: PiStream,
    private readonly host: Host,
    decorateFailure?: (error: Error) => Error,
    /** Launch coordinates, used to probe the runtime for its capabilities. */
    private readonly launchProbe?: { command: string; entry: string },
  ) {
    this.rpc = new PiTransport(
      stream,
      (event) => this.event(event),
      (error) => {
        for (const runs of this.subagentRuns.values())
          void runs.disconnect().catch(() => undefined);
        this.questions.clear();
        this.active?.reject(error);
        this.active?.settled.reject(error);
        for (const request of this.controls.values()) request.reject(error);
        this.controls.clear();
        this.pendingSteer?.applied.reject(error);
      },
      decorateFailure,
    );
  }

  initialize: AgentConnection["initialize"] = async () => {
    await this.rpc.request("get_state");
    await this.rpc.drain();
    if (!this.steerCommand)
      throw new Error("Required Lody Pi extension did not initialize");
    this.capabilities = await this.probe();
    return initializeResponse();
  };

  /** Resolve the runtime's dispatch-reporting capability once per activation. */
  private probe(): Promise<PiCapabilities> {
    if (this.capabilityProbe) return this.capabilityProbe;
    const launch = this.launchProbe;
    if (!launch) return Promise.resolve({ hasDisposition: false });
    this.capabilityProbe = detectCapabilities(
      { command: launch.command, entry: launch.entry },
      {
        ...liveProbes,
        steer: (fields) => this.rpc.request("steer", fields),
      },
    );
    return this.capabilityProbe;
  }

  close(): void {
    this.rpc.close();
  }

  newSession: AgentConnection["newSession"] = async (request) => {
    return this.configure(async () => {
      this.cwd = request.cwd;
      await this.host.configureMcp(request.mcpServers ?? []);
      this.steerCommand = "";
      const result = z
        .object({ cancelled: z.boolean() })
        .parse(await this.rpc.request("new_session"));
      if (result.cancelled)
        throw new Error("Pi extension cancelled session creation");
      await this.initialize({ protocolVersion: 1 });
      return this.prepare(request._meta);
    }, true);
  };

  resumeSession: AgentConnection["resumeSession"] = (request) =>
    this.open(request, false);

  loadSession: AgentConnection["loadSession"] = async (request) => {
    const { configOptions, availableCommands } = await this.open(request, true);
    const response: WithCommands<acp.LoadSessionResponse> = {
      configOptions,
      availableCommands,
    };
    return response;
  };

  private open(
    request: acp.ResumeSessionRequest,
    replay: boolean,
  ): Promise<SessionResponseWithCommands> {
    return this.configure(async () => {
      this.cwd = request.cwd;
      if (
        !path.isAbsolute(request.sessionId) ||
        !request.sessionId.endsWith(".jsonl")
      ) {
        throw new Error(
          "Pi resume requires its native session file; pi-acp ids cannot be migrated automatically",
        );
      }
      // Pi opens a missing path as a new empty session under that name.
      if (!existsSync(request.sessionId))
        throw new Error(`Pi session file not found: ${request.sessionId}`);
      await this.host.configureMcp(request.mcpServers ?? []);
      this.steerCommand = "";
      const result = z.object({ cancelled: z.boolean() }).parse(
        await this.rpc.request("switch_session", {
          sessionPath: request.sessionId,
        }),
      );
      if (result.cancelled)
        throw new Error("Pi extension cancelled session resume");
      await this.initialize({ protocolVersion: 1 });
      const response = await this.prepare(request._meta, replay);
      if (response.sessionId !== request.sessionId)
        throw new Error("Pi resumed a different session file");
      return response;
    }, true);
  }

  private async prepare(
    meta: unknown,
    replay = false,
  ): Promise<SessionResponseWithCommands> {
    const config = startupConfigSchema.safeParse(meta);
    if (config.success) {
      const values = config.data.lody.sessionConfig.configOptionValues;
      // Model first: the supported thinking ladder belongs to the selected model.
      for (const id of ["model", "thinking"]) {
        const value = values[id];
        if (typeof value === "string") await this.setOption(id, value);
      }
    }
    const state = await this.readState();
    if (!state.sessionFile)
      throw new Error("Pi did not provide a persistent session file");
    this.sessionId = state.sessionFile;
    // A new or resumed activation starts a new usage scope from zero.
    this.usageLedger = new SessionUsageAccumulator();
    await this.rpc.drain();
    if (replay) {
      // Replay carries every todo snapshot at its original position instead.
      this.pendingPlan = undefined;
      await this.replayHistory();
    } else if (this.pendingPlan) {
      await this.update({ sessionUpdate: "plan", entries: this.pendingPlan });
      this.pendingPlan = undefined;
    }
    const configOptions = await this.configOptions();
    await this.reportUsage();
    const availableCommands = await this.availableCommands();
    // Lody reads the response top-level field; this keeps live clients in sync.
    setTimeout(() => {
      void this.pushAvailableCommands(availableCommands).catch(() => undefined);
    }, 0);
    return { sessionId: this.sessionId, configOptions, availableCommands };
  }

  /** Replays the native branch Pi will continue from, using Pi's own tool call ids. */
  private async replayHistory(): Promise<void> {
    const { entries, leafId } = entriesSchema.parse(
      await this.rpc.request("get_entries"),
    );
    const byId = new Map(entries.map((entry) => [entry.id, entry]));
    const branch: typeof entries = [];
    for (let id = leafId; id;) {
      const entry = byId.get(id);
      if (!entry) throw new Error(`Pi session entry is missing: ${id}`);
      branch.unshift(entry);
      id = entry.parentId;
    }
    const results = new Map<
      string,
      z.infer<typeof toolResultEntrySchema>["message"]
    >();
    for (const entry of branch) {
      const result = toolResultEntrySchema.safeParse(entry);
      if (result.success)
        results.set(result.data.message.toolCallId, result.data.message);
    }
    for (const entry of branch) {
      const user = userEntrySchema.safeParse(entry);
      if (user.success) {
        for (const content of userContent(
          "message" in user.data
            ? user.data.message.content
            : user.data.content,
        ))
          await this.update({ sessionUpdate: "user_message_chunk", content });
        continue;
      }
      const assistant = assistantEntrySchema.safeParse(entry);
      if (!assistant.success) continue;
      for (const raw of assistant.data.message.content) {
        const parsed = assistantBlockSchema.safeParse(raw);
        if (!parsed.success) continue;
        const block = parsed.data;
        if (block.type === "text" && block.text)
          await this.update({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: block.text },
          });
        else if (block.type === "thinking" && block.thinking)
          await this.update({
            sessionUpdate: "agent_thought_chunk",
            content: { type: "text", text: block.thinking },
          });
        else if (block.type === "toolCall") {
          const result = results.get(block.id);
          await this.update({
            ...this.toolCall(
              {
                toolCallId: block.id,
                toolName: block.name,
                args: block.arguments,
                result,
              },
              result && !result.isError ? "completed" : "failed",
            ),
            sessionUpdate: "tool_call",
          });
          const todos = todosSchema.safeParse(result?.details);
          if (block.name === "todo" && todos.success)
            await this.update({
              sessionUpdate: "plan",
              entries: planEntries(todos.data),
            });
        }
      }
    }
  }

  private async configOptions(
    observedState?: z.infer<typeof stateSchema>,
  ): Promise<acp.SessionConfigOption[]> {
    const [rawState, rawModels, rawLevels] = await Promise.all([
      observedState ?? this.readState(),
      this.rpc.request("get_available_models"),
      this.rpc.request("get_available_thinking_levels"),
    ]);
    const state = rawState;
    if (!state.sessionFile)
      throw new Error("Pi did not provide a persistent session file");
    const models = z
      .object({ models: z.array(modelSchema) })
      .parse(rawModels).models;
    const levels = z
      .object({ levels: z.array(z.string()) })
      .parse(rawLevels).levels;
    if (!state.model || models.length === 0)
      throw new Error(
        "Pi has no configured model. Run pi /login on this machine or add provider API keys to the agent environment.",
      );
    return [
      ...(state.model
        ? [
            {
              id: "model",
              name: "Model",
              category: "model",
              type: "select" as const,
              currentValue: `${state.model.provider}/${state.model.id}`,
              options: models.map((model) => ({
                value: `${model.provider}/${model.id}`,
                name: `${model.name} (${model.provider})`,
              })),
            },
          ]
        : []),
      {
        id: "thinking",
        name: "Thinking",
        category: "thought_level",
        type: "select",
        currentValue: state.thinkingLevel,
        options: levels.map((value) => ({ value, name: value })),
      },
    ];
  }

  private async setOption(id: string, value: string): Promise<void> {
    if (id === "model") {
      const models = z
        .object({ models: z.array(modelSchema) })
        .parse(await this.rpc.request("get_available_models")).models;
      const model = models.find(
        (candidate) => `${candidate.provider}/${candidate.id}` === value,
      );
      if (!model) throw new Error(`Pi model is unavailable: ${value}`);
      await this.rpc.request("set_model", {
        provider: model.provider,
        modelId: model.id,
      });
    } else if (id === "thinking") {
      const { levels } = z
        .object({ levels: z.array(z.string()) })
        .parse(await this.rpc.request("get_available_thinking_levels"));
      if (!levels.includes(value))
        throw new Error(
          `Pi thinking level is unavailable for this model: ${value}`,
        );
      await this.rpc.request("set_thinking_level", { level: value });
    } else throw new Error(`Unsupported Pi configuration option: ${id}`);
  }

  setSessionConfigOption: AgentConnection["setSessionConfigOption"] = async (
    request,
  ) => {
    this.assertSession(request.sessionId);
    return this.configure(async () => {
      await this.readState();
      if (typeof request.value !== "string")
        throw new Error("Pi configuration requires a select value");
      await this.setOption(request.configId, request.value);
      const configOptions = await this.configOptions();
      await this.reportUsage();
      return { configOptions };
    });
  };

  prompt: AgentConnection["prompt"] = async (request) => {
    if (this.active) await this.waitForSettledRun();
    this.assertSession(request.sessionId);
    this.assertIdle();
    const { message, images } = this.promptContent(request.prompt);
    if (message.startsWith(`/${this.steerCommand}`))
      throw new Error(
        "Internal Pi control commands cannot be submitted as user prompts",
      );
    await this.rejectExtensionCommand(message);
    const run: Run = {
      ...deferred<acp.PromptResponse>(),
      activities: new Map(),
      finished: deferred<void>(),
      started: false,
      modelSettled: false,
      completed: false,
      cancelled: false,
      settled: deferred<void>(),
      message,
    };
    this.active = run;
    try {
      await this.readState();
      await this.rpc.drain();
      this.assertReady();
      if (run.cancelled) {
        await run.cancellation;
        this.finishRun(run);
        return await run.promise;
      }
      let responseText: string | undefined;
      if (message.trim() === "/stats" && images.length === 0) {
        const stats = await this.reportUsage();
        if (!stats) run.error = "Pi session usage is unavailable";
        if (stats)
          responseText = `Pi session usage: ${stats.tokens.input} input, ${stats.tokens.output} output, ${stats.tokens.cacheRead} cache read, ${stats.tokens.cacheWrite} cache write tokens; $${stats.cost.toFixed(6)}.`;
      } else if (
        /^\/compact(?:\s|$)/.test(message.trim()) &&
        images.length === 0
      ) {
        try {
          await this.rpc.request("compact", {
            customInstructions:
              message.trim().slice("/compact".length).trim() || undefined,
          });
        } catch (error) {
          run.error = error instanceof Error ? error.message : String(error);
        }
        responseText = "Context compacted.";
      } else {
        run.accepted = this.rpc.request("prompt", {
          message,
          ...(images.length ? { images } : {}),
        });
        await run.accepted;
        if (this.legacy) {
          await run.settled.promise;
        } else {
          const disposition = await this.promptDisposition(run);
          run.disposition = disposition;
          if (disposition && disposition !== "started") {
            await this.awaitPossibleDetachedTurn(run);
            if (run.started) await run.settled.promise;
          } else {
            await run.settled.promise;
          }
        }
      }
      // Execution and output delivery are separate barriers.
      await this.rpc.drain();
      this.assertReady();
      run.completed = true;
      const pending = this.pendingSteer;
      if (pending && !run.cancelled) {
        await this.rpc.request("clear_queue");
        await this.rpc.drain();
        if (this.pendingSteer === pending && !run.cancelled)
          pending.applied.reject(
            acp.RequestError.invalidRequest(
              undefined,
              "Pi settled before steer delivery",
            ),
          );
      }
      await this.refreshConfigOptions();
      if (message.trim() !== "/stats" || images.length)
        await this.reportUsage();
      if (!run.cancelled && !run.error && responseText)
        await this.update({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: responseText },
        });
      this.finishRun(run);
      return await run.promise;
    } finally {
      await run.cancellation?.catch(() => undefined);
      for (const key of run.activities.keys())
        await this.endActivity(
          run,
          key,
          run.cancelled ? "Cancelled" : "Pi ended without an activity result",
        ).catch(() => undefined);
      await this.cancelQuestions().catch(() => undefined);
      if (this.active === run) this.active = undefined;
      run.finished.resolve();
    }
  };

  private finishRun(run: Run): void {
    if (run.cancelled) run.resolve({ stopReason: "cancelled" });
    else if (run.error) run.reject(new Error(run.error));
    else run.resolve({ stopReason: run.stopReason ?? "end_turn" });
  }

  private async refreshConfigOptions(): Promise<void> {
    // Identity/transport failures remain fatal. Only the display refresh is optional.
    const state = await this.readState();
    let configOptions: acp.SessionConfigOption[];
    try {
      configOptions = await this.configOptions(state);
    } catch (error) {
      this.rpc.assertOpen();
      process.stderr.write(
        `Pi configuration refresh failed: ${String(error)}\n`,
      );
      return;
    }
    await this.update({ sessionUpdate: "config_option_update", configOptions });
  }

  cancel: AgentConnection["cancel"] = async (request) => {
    this.assertSession(request.sessionId);
    const run = this.active;
    if (!run) return;
    run.cancelled = true;
    await this.cancelRun(run);
    await run.finished.promise;
  };

  private cancelRun(run: Run): Promise<void> {
    // Native abort cannot cancel a model run that has not passed preflight yet.
    // Our packaged tools never ask questions or dispatch work during preflight.
    run.cancellation ??= (async () => {
      await run.accepted?.catch(() => undefined);
      await this.abort();
    })();
    return run.cancellation;
  }

  private async abort(): Promise<void> {
    const pending = this.pendingSteer;
    try {
      await this.cancelQuestions();
      // clear_queue alone cannot see messages already drained into Pi's agent loop.
      // abort waits for idle; drain observes their message_start before classifying delivery.
      await this.rpc.request("clear_queue");
      await this.rpc.request("abort");
      await this.rpc.drain();
      if (pending && this.pendingSteer === pending)
        pending.applied.reject(
          acp.RequestError.invalidRequest(
            undefined,
            "Pi steer cancelled before delivery",
          ),
        );
    } catch (error) {
      pending?.applied.reject(
        error instanceof Error ? error : new Error("Pi cancellation failed"),
      );
      throw error;
    }
  }

  private promptContent(blocks: acp.ContentBlock[]) {
    const text: string[] = [];
    const images: Array<{ type: "image"; data: string; mimeType: string }> = [];
    for (const block of blocks) {
      switch (block.type) {
        case "text":
          text.push(block.text);
          break;
        case "image":
          images.push({
            type: "image",
            data: block.data,
            mimeType: block.mimeType,
          });
          break;
        case "resource_link":
          text.push(`${block.name}: ${block.uri}`);
          break;
        case "resource":
          if ("text" in block.resource)
            text.push(`${block.resource.uri}\n${block.resource.text}`);
          else
            throw new Error(
              "Pi does not support embedded binary resources; attach an image or a file link",
            );
          break;
        default:
          throw new Error(`Pi does not support prompt content: ${block.type}`);
      }
    }
    return { message: text.join("\n\n"), images };
  }

  private update(update: acp.SessionNotification["update"]): Promise<void> {
    if (!this.sessionId || !this.steerCommand) return Promise.resolve();
    return this.host.update({ sessionId: this.sessionId, update });
  }

  private notice(
    message: string,
    level: "info" | "warning" | "error" = "info",
  ) {
    return this.update({
      sessionUpdate: "session_info_update",
      _meta: { lody: { notice: { level, message, source: "pi" } } },
    });
  }

  private observeSession(file: string | undefined): void {
    if (this.sessionId && file !== this.sessionId) {
      this.sessionId = "";
      this.steerCommand = "";
      const error = new Error(
        "Pi changed its native session; explicitly create or resume a session",
      );
      this.active?.reject(error);
      this.pendingSteer?.applied.reject(error);
      throw error;
    }
  }

  private async readState() {
    const state = stateSchema.parse(await this.rpc.request("get_state"));
    this.observeSession(state.sessionFile);
    return state;
  }

  private assertReady(): void {
    if (!this.steerCommand)
      throw new Error("Required Lody Pi extension did not initialize");
  }

  private async event(event: Record<string, unknown>): Promise<void> {
    if (
      event.type === "extension_ui_request" &&
      event.method === "notify" &&
      typeof event.message === "string" &&
      event.message.startsWith("lody-rpc:")
    ) {
      const payload = z
        .object({ type: z.string() })
        .passthrough()
        .parse(JSON.parse(event.message.slice("lody-rpc:".length)));
      if (payload.type === "lody_control") {
        const response = this.controls.get(String(payload.id));
        if (payload.error) response?.reject(new Error(String(payload.error)));
        else response?.resolve(payload.data);
        return;
      }
      if (payload.type === "lody_todos") {
        const entries = planEntries(todosSchema.parse(payload));
        if (!this.sessionId) this.pendingPlan = entries;
        else await this.update({ sessionUpdate: "plan", entries });
        return;
      }
      if (payload.type === "lody_usage") {
        const usage = z
          .object({ taskId: z.string(), message: piAssistantUsageSchema })
          .safeParse(payload);
        if (usage.success)
          this.recordUsage(
            `subagent:${usage.data.taskId}:${piResponseOperation(usage.data.message)}`,
            piModelKey(usage.data.message.provider, usage.data.message.model),
            usage.data.message.usage,
          );
        return;
      }
      if (payload.type === "lody_subagent") {
        const task = z
          .object({
            taskId: z.string(),
            status: z.enum(["in_progress", "completed", "failed"]),
            description: z.string(),
          })
          .passthrough()
          .parse(payload.task);
        if (this.host.subagentEvents && this.sessionId) {
          const runs = this.subagents();
          if (payload.event === "started") {
            await runs.start(task.taskId, {
              state: "running",
              parentRunId: null,
              description: task.description,
              ...(typeof task.parentToolCallId === "string"
                ? { parentToolCallId: task.parentToolCallId }
                : {}),
              ...(typeof task.modelId === "string"
                ? { modelId: task.modelId }
                : {}),
              ...(typeof task.startedAtEpochSeconds === "number"
                ? { startedAtEpochSeconds: task.startedAtEpochSeconds }
                : {}),
              support: {
                stream: ["text", "thought", "tool"],
                progress: true,
                outputRead: "none",
                cancel: false,
              },
            });
          }
          if (typeof task.lastToolName === "string")
            await runs.progress(task.taskId, {
              lastToolName: task.lastToolName,
            });
          if (task.status !== "in_progress")
            await runs.snapshot(task.taskId, {
              state: payload.state === "killed" ? "cancelled" : task.status,
              ...(typeof task.endedAtEpochSeconds === "number"
                ? { endedAtEpochSeconds: task.endedAtEpochSeconds }
                : {}),
              ...(typeof task.error === "string"
                ? {
                    reason: {
                      code: payload.state === "killed" ? "cancelled" : "error",
                      message: task.error,
                    },
                  }
                : {}),
            });
          return;
        }
        await this.update({
          sessionUpdate:
            payload.event === "started" ? "tool_call" : "tool_call_update",
          toolCallId: task.taskId,
          title: task.description,
          status: task.status,
          _meta: { lody: { task } },
        });
        return;
      }
      if (payload.type === "lody_subagent_output") {
        if (
          !this.host.subagentEvents ||
          !this.sessionId ||
          typeof payload.taskId !== "string"
        )
          return;
        const childEvent = z
          .record(z.string(), z.unknown())
          .parse(payload.event);
        const runs = this.subagents();
        if (childEvent.type === "message_update") {
          const delta = z
            .object({ type: z.string(), delta: z.string().optional() })
            .parse(childEvent.assistantMessageEvent);
          if (delta.type === "text_delta" || delta.type === "thinking_delta")
            await runs.output(
              payload.taskId,
              {
                sessionUpdate:
                  delta.type === "text_delta"
                    ? "agent_message_chunk"
                    : "agent_thought_chunk",
                content: { type: "text", text: delta.delta ?? "" },
              },
              typeof payload.messageId === "string"
                ? { messageId: payload.messageId }
                : {},
            );
        } else {
          await this.tool(childEvent, payload.taskId);
        }
        return;
      }
      event = z
        .object({
          type: z.enum([
            "lody_steer_ready",
            "lody_steer_refused",
            "lody_not_ready",
          ]),
          version: z.number().optional(),
          steerId: z.string().optional(),
          command: z.string().optional(),
          sessionFile: z.string().optional(),
        })
        .parse(JSON.parse(event.message.slice("lody-rpc:".length)));
    }
    if (event.type === "lody_steer_ready") {
      // Lifecycle events precede any output from the replacement runtime.
      try {
        this.observeSession(
          typeof event.sessionFile === "string" ? event.sessionFile : undefined,
        );
      } catch {
        return;
      }
      this.steerCommand =
        event.version === 1 &&
        typeof event.command === "string" &&
        /^[a-zA-Z0-9_-]+$/.test(event.command)
          ? event.command
          : "";
      return;
    }
    if (event.type === "lody_not_ready") {
      this.steerCommand = "";
      this.pendingSteer?.applied.reject(
        new Error("Pi extension runtime stopped"),
      );
      await this.cancelQuestions();
      return;
    }
    if (event.type === "lody_steer_refused") {
      if (this.pendingSteer && event.steerId === this.pendingSteer.id)
        this.pendingSteer.applied.reject(
          acp.RequestError.invalidRequest(
            undefined,
            "Pi is idle; steer was not delivered",
          ),
        );
      return;
    }
    if (event.type === "extension_ui_request") {
      if (event.method === "notify") {
        const notice = z
          .object({
            message: z.string(),
            notifyType: z.enum(["info", "warning", "error"]).optional(),
          })
          .parse(event);
        await this.notice(notice.message, notice.notifyType);
        return;
      }
      const request = questionSchema.parse(event);
      // Do not block the wire/notification queue on a human response.
      void this.question(request).catch(() => undefined);
      return;
    }
    const run = this.active;
    if (event.type === "extension_error" && (!run || run.completed)) {
      process.stderr.write(`Pi extension diagnostic: ${String(event.error)}\n`);
      return;
    }
    // An extension can dispatch its run after the handler returned and the
    // adapter already closed the turn as idle. Report its arrival; its output
    // is not forwarded, because no run is left to carry it to the client.
    if (!run) {
      const closed = this.closedIdleRun;
      if (closed && event.type === "agent_start") {
        this.closedIdleRun = undefined;
        void this.notice(
          `Pi started a model turn after /${closed.message.trim().slice(1).split(/\s+/)[0] || closed.message} was handled.`,
        );
      }
      return;
    }
    switch (event.type) {
      case "agent_start":
        run.started = true;
        run.modelSettled = false;
        break;
      case "agent_settled":
        run.modelSettled = true;
        run.settled.resolve();
        break;
      case "message_start": {
        const message = z
          .object({
            role: z.string(),
            customType: z.string().optional(),
            details: z.unknown().optional(),
          })
          .parse(event.message);
        const pending = this.pendingSteer;
        const identity = z
          .object({ steerId: z.string() })
          .safeParse(message.details);
        if (
          pending &&
          pending.run === run &&
          message.role === "custom" &&
          message.customType === "lody-steer" &&
          identity.success &&
          identity.data.steerId === pending.id
        ) {
          this.pendingSteer = undefined;
          pending.applied.resolve();
          // The existing host lease switches history ownership before any subsequent output.
          await this.host.extension(
            LODY_EXTENSION_METHODS.sessionSteerApplied,
            {
              sessionId: this.sessionId,
              steerId: pending.id,
            },
          );
        }
        if (message.role === "assistant") {
          run.error = undefined;
          run.stopReason = undefined;
          await this.readState();
        }
        break;
      }
      case "message_update": {
        const delta = z
          .object({ type: z.string(), delta: z.string().optional() })
          .parse(event.assistantMessageEvent);
        if (
          (delta.type === "text_delta" || delta.type === "thinking_delta") &&
          delta.delta
        ) {
          await this.update({
            sessionUpdate:
              delta.type === "text_delta"
                ? "agent_message_chunk"
                : "agent_thought_chunk",
            content: { type: "text", text: delta.delta },
          });
        }
        break;
      }
      case "message_end": {
        const message = z
          .object({
            role: z.string(),
            stopReason: z.string().optional(),
            errorMessage: z.string().optional(),
          })
          .parse(event.message);
        if (message.role !== "assistant") break;
        // Errored and aborted responses still spent the tokens they report.
        const response = piAssistantUsageSchema.safeParse(event.message);
        if (response.success) {
          this.lastModelKey = piModelKey(
            response.data.provider,
            response.data.model,
          );
          this.recordUsage(
            `message:${piResponseOperation(response.data)}`,
            this.lastModelKey,
            response.data.usage,
          );
        }
        if (message.stopReason === "error")
          run.error = message.errorMessage ?? "Pi model request failed";
        if (message.stopReason === "aborted") run.cancelled = true;
        run.stopReason =
          message.stopReason === "length" ? "max_tokens" : undefined;
        // Pi owns validity, compaction boundaries and the matching context window.
        // Refresh between assistant messages as well as at final settlement.
        await this.reportUsage();
        break;
      }
      case "tool_execution_start":
      case "tool_execution_update":
      case "tool_execution_end":
        await this.tool(event);
        break;
      case "compaction_start":
        await this.startActivity(run, "compaction", "Compact context", {
          version: 1,
          kind: "context_compaction",
          automatic: event.reason !== "manual",
        });
        break;
      case "compaction_end": {
        const result = z
          .object({
            tokensBefore: z.number().nonnegative().optional(),
            estimatedTokensAfter: z.number().nonnegative().optional(),
            usage: piUsageSchema.optional(),
          })
          .optional()
          .parse(event.result);
        // Pi reports the summary call's usage without its model; it runs on the
        // session model: the latest response's, else the configured one.
        if (result?.usage) {
          const model =
            this.lastModelKey ??
            (await this.readState().then(
              (state) =>
                state.model && piModelKey(state.model.provider, state.model.id),
              () => undefined,
            ));
          if (model)
            this.recordUsage(
              `compaction:${++this.compactions}`,
              model,
              result.usage,
            );
        }
        await this.endActivity(
          run,
          "summaryRetry",
          event.aborted ? "Cancelled" : undefined,
        );
        await this.endActivity(
          run,
          "compaction",
          event.aborted
            ? "Cancelled"
            : typeof event.errorMessage === "string"
              ? event.errorMessage
              : !result
                ? "Compaction did not produce a summary"
                : undefined,
          {
            usedTokensBefore: result?.tokensBefore,
            usedTokensAfter: result?.estimatedTokensAfter,
          },
        );
        if (!run.started) {
          if (event.aborted) run.cancelled = true;
          else if (typeof event.errorMessage === "string")
            run.error = event.errorMessage;
        }
        break;
      }
      case "auto_retry_start":
      case "summarization_retry_scheduled":
        await this.startActivity(
          run,
          event.type === "auto_retry_start" ? "retry" : "summaryRetry",
          `${event.type === "auto_retry_start" ? "Model retry" : "Summary retry wait"} ${event.attempt}/${event.maxAttempts}`,
          { version: 1, kind: "retry", automatic: true },
        );
        break;
      case "auto_retry_end":
        await this.endActivity(
          run,
          "retry",
          event.success
            ? undefined
            : String(event.finalError ?? "Retry failed"),
        );
        break;
      case "summarization_retry_attempt_start":
      case "summarization_retry_finished":
        // This event ends the backoff, not the enclosing compaction operation.
        await this.endActivity(
          run,
          "summaryRetry",
          run.cancelled ? "Cancelled" : undefined,
        );
        break;
      case "extension_error": {
        const diagnostic = z
          .object({ event: z.string(), error: z.string() })
          .parse(event);
        // Pi acknowledges a handled command even when its handler throws. Ordinary
        // extension callbacks are diagnostics, not authority over the model result.
        if (diagnostic.event === "command" && !run.started) {
          run.error = diagnostic.error;
        } else {
          await this.notice(diagnostic.error, "warning");
        }
        break;
      }
    }
  }

  private async startActivity(
    run: Progress,
    key: string,
    title: string,
    meta: LodyActivityMeta,
  ) {
    const existing = run.activities.get(key);
    const activity = existing ?? { id: `pi-${key}-${randomUUID()}`, meta };
    run.activities.set(key, activity);
    await this.update({
      sessionUpdate: existing ? "tool_call_update" : "tool_call",
      toolCallId: activity.id,
      title,
      kind: "other",
      status: "in_progress",
      _meta: { lody: { activity: meta } },
    });
  }

  private async endActivity(
    run: Progress,
    key: string,
    failureReason?: string,
    details: Partial<LodyActivityMeta> = {},
  ) {
    const activity = run.activities.get(key);
    if (!activity) return;
    run.activities.delete(key);
    await this.update({
      sessionUpdate: "tool_call_update",
      toolCallId: activity.id,
      status: failureReason ? "failed" : "completed",
      _meta: {
        lody: {
          activity: {
            ...activity.meta,
            ...details,
            ...(failureReason ? { failureReason } : {}),
          },
        },
      },
    });
  }

  private recordUsage(
    operationId: string,
    model: string,
    usage: z.infer<typeof piUsageSchema>,
  ): void {
    if (!this.sessionId) return;
    const update = this.usageLedger.update(this.sessionId, operationId, {
      [model]: toModelUsage(usage),
    });
    if (update) this.host.usage(update);
  }

  /** The dispatch disposition the runtime attached to the accepted prompt. */
  private async promptDisposition(run: Run): Promise<string | undefined> {
    const payload = (await run.accepted) as
      { disposition?: string } | undefined;
    return typeof payload?.disposition === "string"
      ? payload.disposition
      : undefined;
  }

  /**
   * A handled prompt started no run of its own, but an extension may still
   * dispatch one after its handler returns, as pi-goal-x does for autonomous
   * goals. Hold the turn until the runtime stays quiet, measured from the last
   * sign of work so a late dispatch cannot find the turn already closed.
   */
  private async awaitPossibleDetachedTurn(run: Run): Promise<void> {
    await this.rpc.drain();
    this.assertReady();
    if (run.started || (await this.turnUnderway(run))) return;
    const opened = Date.now();
    let lastBusy = opened;
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, DETACHED_POLL_MS));
      await this.rpc.drain();
      this.assertReady();
      // A dispatched run hands the wait back to the caller's settlement barrier.
      if (run.started || run.cancelled) return;
      if (await this.turnUnderway(run)) lastBusy = Date.now();
      if (Date.now() - lastBusy >= DETACHED_QUIET_MS) break;
      if (Date.now() - opened >= DETACHED_MAX_WINDOW_MS) break;
    }
    if (run.started || (await this.turnUnderway(run))) return;
    // Remembered so a run the extension dispatched after the window closed is
    // reported rather than silently dropped into an idle session.
    this.closedIdleRun = run;
    await this.notice(
      `Pi handled /${run.message.trim().slice(1).split(/\s+/)[0] || run.message} without starting a model turn.`,
    );
  }

  /**
   * Pi's own view of whether work is in flight, not just what arrived.
   *
   * A settled run reads as busy on purpose: it hands the wait back to the
   * caller, whose settlement barrier is already satisfied.
   */
  private async turnUnderway(run: Run): Promise<boolean> {
    if (run.started || run.modelSettled) return true;
    try {
      const state = await this.readState();
      return Boolean(
        state.isStreaming || state.isCompacting || state.pendingMessageCount,
      );
    } catch {
      // An unreadable state must not read as idle, or flapping I/O would let the
      // window close a turn that is still running.
      return true;
    }
  }

  /** Local handlers win over Pi's own commands on a name collision. */
  private async availableCommands(): Promise<acp.AvailableCommand[]> {
    const commands = new Map<string, acp.AvailableCommand>();
    for (const command of LOCAL_COMMANDS)
      commands.set(command.name, { ...command, input: null });
    try {
      // get_commands is a Pi 0.87.0 RPC snapshot; a shape change degrades
      // silently to the local commands below.
      const result = commandsSchema.parse(
        await this.rpc.request("get_commands"),
      );
      for (const command of result.commands) {
        // The private steer transport is registered as a Pi command, but it is
        // adapter plumbing and must never reach the client command list.
        if (command.name === this.steerCommand) continue;
        // Without a disposition a command's model turn is unknowable.
        if (this.legacy && command.source === "extension") continue;
        if (commands.has(command.name)) continue;
        commands.set(command.name, {
          name: command.name,
          description: command.description ?? "",
          input: null,
        });
      }
    } catch {
      // A missing command source is not fatal.
    }
    return Array.from(commands.values());
  }

  /** Reject extension command prompts by source before a run is created. */
  private async rejectExtensionCommand(message: string): Promise<void> {
    const trimmed = message.trim();
    if (!trimmed.startsWith("/")) return;
    const name = trimmed.slice(1).split(/\s+/)[0];
    if (!name || name === this.steerCommand) return;
    for (const local of LOCAL_COMMANDS) if (local.name === name) return;
    try {
      const result = commandsSchema.parse(
        await this.rpc.request("get_commands"),
      );
      const command = result.commands.find((entry) => entry.name === name);
      if (this.legacy && command?.source === "extension")
        throw new ExtensionCommandError(name);
    } catch (error) {
      if (error instanceof ExtensionCommandError) throw error;
      // A missing command source must not block unrelated prompts.
    }
  }

  private async pushAvailableCommands(
    commands: acp.AvailableCommand[],
  ): Promise<void> {
    await this.update({
      sessionUpdate: "available_commands_update",
      availableCommands: commands,
    });
  }

  /** Context occupancy and the /status text come from Pi stats; accounting does not. */
  private async reportUsage() {
    const sessionId = this.sessionId;
    const parsed = statsSchema.safeParse(
      await this.rpc.request("get_session_stats").catch(() => {
        // An unavailable snapshot is optional; a failed connection is not,
        // including startup/configuration where there is no active prompt to reject.
        this.rpc.assertOpen();
        return undefined;
      }),
    );
    if (
      !parsed.success ||
      !sessionId ||
      sessionId !== this.sessionId ||
      !this.steerCommand
    )
      return;
    const { contextUsage } = parsed.data;
    if (contextUsage?.tokens != null)
      await this.update({
        sessionUpdate: "usage_update",
        size: contextUsage.contextWindow,
        used: contextUsage.tokens,
      });
    return parsed.data;
  }

  private async tool(
    event: Record<string, unknown>,
    childId?: string,
  ): Promise<void> {
    const tool = z
      .object({
        type: z.string(),
        toolCallId: z.string(),
        toolName: z.string(),
        args: z.record(z.string(), z.unknown()).optional(),
        result: z
          .object({ content: contentSchema, details: z.unknown().optional() })
          .optional(),
        partialResult: z.object({ content: contentSchema }).optional(),
        isError: z.boolean().optional(),
      })
      .parse(event);
    if (!childId && this.host.subagentEvents && tool.toolName === "subagent")
      return;
    const start = tool.type === "tool_execution_start";
    const end = tool.type === "tool_execution_end";
    const notification = this.toolCall(
      tool,
      end ? (tool.isError ? "failed" : "completed") : "in_progress",
    );
    const publish = childId
      ? (update: acp.SessionNotification["update"]) =>
          this.subagents().output(childId, update)
      : (update: acp.SessionNotification["update"]) => this.update(update);
    await publish(
      start
        ? { ...notification, sessionUpdate: "tool_call" }
        : { ...notification, sessionUpdate: "tool_call_update" },
    );
  }

  private toolCall(
    tool: {
      toolCallId: string;
      toolName: string;
      args?: Record<string, unknown>;
      result?: { content: z.infer<typeof contentSchema>; details?: unknown };
      partialResult?: { content: z.infer<typeof contentSchema> };
    },
    status: acp.ToolCallStatus,
  ): acp.ToolCall {
    const args = tool.args ?? {};
    const file =
      typeof args.path === "string"
        ? path.resolve(this.cwd, args.path)
        : undefined;
    const kind: acp.ToolKind =
      tool.toolName === "bash"
        ? "execute"
        : tool.toolName === "read"
          ? "read"
          : ["edit", "write"].includes(tool.toolName)
            ? "edit"
            : ["grep", "find", "ls"].includes(tool.toolName)
              ? "search"
              : "other";
    const content = tool.result?.content ?? tool.partialResult?.content;
    return {
      toolCallId: tool.toolCallId,
      title: tool.toolName,
      kind,
      status,
      ...(tool.args
        ? {
            rawInput: {
              ...args,
              ...(file ? { file_path: file } : {}),
              ...(typeof args.oldText === "string"
                ? { old_string: args.oldText }
                : {}),
              ...(typeof args.newText === "string"
                ? { new_string: args.newText }
                : {}),
            },
          }
        : {}),
      ...(file ? { locations: [{ path: file }] } : {}),
      ...(tool.result ? { rawOutput: tool.result } : {}),
      ...(content
        ? {
            content: content.map((block) => ({
              type: "content" as const,
              content: block,
            })),
          }
        : {}),
    };
  }

  private async question(
    request: z.infer<typeof questionSchema>,
  ): Promise<void> {
    const run = this.active;
    const cancel = () =>
      this.rpc.send({
        type: "extension_ui_response",
        id: request.id,
        cancelled: true,
      });
    if (
      request.method !== "input" ||
      !request.title?.startsWith(QUESTION_PREFIX) ||
      !run ||
      run.completed ||
      run.cancelled ||
      !this.sessionId ||
      !this.steerCommand
    )
      return cancel();
    const respond = async (fields: Record<string, unknown>): Promise<void> => {
      if (!this.questions.delete(request.id)) return;
      await this.rpc.send({
        type: "extension_ui_response",
        id: request.id,
        ...fields,
      });
    };
    this.questions.set(request.id, () => respond({ cancelled: true }));
    try {
      const form = z
        .object({ toolCallId: z.string(), questions: questionsSchema })
        .parse(JSON.parse(request.title.slice(QUESTION_PREFIX.length)));
      const properties: Extract<
        acp.CreateElicitationRequest,
        { mode?: "form" }
      >["requestedSchema"]["properties"] = {};
      for (const [index, question] of form.questions.entries()) {
        const key = `q${index}`;
        properties[key] = {
          type: "string",
          title: question.header,
          description: question.question,
          ...(question.options.length
            ? {
                oneOf: question.options.map((option) => ({
                  const: option.label,
                  title: option.label,
                  ...(option.description
                    ? { description: option.description }
                    : {}),
                })),
              }
            : {}),
        };
        if (question.options.length && question.allowCustomAnswer)
          properties[`${key}_custom`] = {
            type: "string",
            _meta: {
              lody: { elicitation: { version: 1, customAnswerFor: key } },
            },
          };
      }
      const response = await this.host.question({
        mode: "form",
        sessionId: this.sessionId,
        toolCallId: form.toolCallId,
        message:
          form.questions.length === 1
            ? form.questions[0].question
            : "Pi needs your input",
        requestedSchema: { type: "object", properties },
        _meta: {
          lody: {
            elicitation: { version: 1, autoResolveAfterSeconds: null },
          },
        },
      });
      if (
        this.active !== run ||
        run.cancelled ||
        run.completed ||
        response.action !== "accept"
      )
        return respond({ cancelled: true });
      const content = z.record(z.string(), z.unknown()).parse(response.content);
      const answers: Record<string, string> = {};
      for (const [index, question] of form.questions.entries()) {
        const custom = content[`q${index}_custom`];
        const value =
          question.allowCustomAnswer &&
          typeof custom === "string" &&
          custom.trim()
            ? custom
            : content[`q${index}`];
        if (
          typeof value !== "string" ||
          !value.trim() ||
          (!question.allowCustomAnswer &&
            question.options.length &&
            !question.options.some((option) => option.label === value))
        )
          return respond({ cancelled: true });
        answers[question.id] = value;
      }
      return respond({ value: JSON.stringify(answers) });
    } catch {
      await respond({ cancelled: true });
    }
  }

  private async cancelQuestions(): Promise<void> {
    await Promise.all([...this.questions.values()].map((cancel) => cancel()));
  }

  private async configure<T>(
    action: () => Promise<T>,
    replaceSession = false,
  ): Promise<T> {
    await this.waitForSettledRun();
    this.assertIdle();
    this.configuring = true;
    // A replacement can change Pi's durable identity before its ACK or config succeeds.
    // Never let an old ACP id address that new file, including after failure.
    if (replaceSession) this.sessionId = "";
    try {
      return await action();
    } catch (error) {
      if (replaceSession) this.sessionId = "";
      throw error;
    } finally {
      this.configuring = false;
    }
  }

  private assertIdle(): void {
    if (this.active || this.configuring)
      throw new Error(
        "Pi already has an active prompt or configuration operation",
      );
  }
  private async waitForSettledRun(): Promise<void> {
    const run = this.active;
    // A new input after model output stops may wait for callback cleanup. This
    // progress flag never completes the existing request or releases admission.
    if (run?.modelSettled || run?.completed || run?.cancelled)
      await run.finished.promise;
  }
  private assertSession(id: string): void {
    if (!this.sessionId || id !== this.sessionId)
      throw new Error("Pi session does not match the active session");
  }

  request: AgentConnection["request"] = async <T>(
    method: string,
    params?: unknown,
  ): Promise<T> => {
    const operation =
      method === LODY_EXTENSION_METHODS.subagentsList
        ? "list"
        : method === LODY_EXTENSION_METHODS.subagentsCancel
          ? "cancel"
          : method === LODY_EXTENSION_METHODS.subagentsOutput
            ? "output"
            : undefined;
    if (operation) {
      const input = z
        .object({
          sessionId: z.string(),
          taskId: z.string().optional(),
          activeOnly: z.boolean().optional(),
          tail: z.number().int().positive().optional(),
        })
        .parse(params);
      this.assertSession(input.sessionId);
      this.assertReady();
      const id = randomUUID();
      const response = deferred<unknown>();
      this.controls.set(id, response);
      try {
        await this.rpc.request("prompt", {
          message:
            `/${this.steerCommand} ` +
            JSON.stringify({ ...input, op: operation, id }),
        });
        return (await response.promise) as T;
      } finally {
        this.controls.delete(id);
      }
    }
    if (method !== LODY_EXTENSION_METHODS.sessionSteer)
      throw new Error("Pi does not implement this ACP extension");
    const request = z
      .object({
        sessionId: z.string(),
        steerId: z.string(),
        prompt: z.array(
          z.union([
            z.object({ type: z.literal("text"), text: z.string() }),
            z.object({
              type: z.literal("image"),
              data: z.string(),
              mimeType: z.string(),
            }),
            z.object({
              type: z.literal("resource_link"),
              uri: z.string(),
              name: z.string(),
            }),
            z.object({
              type: z.literal("resource"),
              resource: z.object({ uri: z.string(), text: z.string() }),
            }),
          ]),
        ),
      })
      .parse(params);
    this.assertSession(request.sessionId);
    this.assertReady();
    const run = this.active;
    if (
      !run ||
      !run.started ||
      run.completed ||
      run.cancelled ||
      this.pendingSteer
    )
      throw acp.RequestError.invalidRequest(
        undefined,
        "No available Pi turn for steer",
      );
    const { message, images } = this.promptContent(request.prompt);
    const pending = {
      id: request.steerId,
      run,
      applied: deferred<void>(),
    };
    this.pendingSteer = pending;
    try {
      // The extension preserves identity in Pi metadata and atomically injects or refuses.
      await this.rpc.request("prompt", {
        streamingBehavior: "steer",
        message:
          `/${this.steerCommand} ` +
          JSON.stringify({
            steerId: request.steerId,
            content: [{ type: "text", text: message }, ...images],
          }),
      });
      await pending.applied.promise;
      return { outcome: "injected" } as T;
    } finally {
      if (this.pendingSteer === pending) this.pendingSteer = undefined;
    }
  };
}

export function initializeResponse(): acp.InitializeResponse {
  return {
    protocolVersion: 1,
    agentInfo: { name: "pi-rpc", version: PI_RPC_VERSION },
    agentCapabilities: {
      _meta: {
        lody: {
          usage: { version: 1 },
          subagentEvents: { version: 1 },
          compaction: { version: 1 },
          subagents: {
            version: 1,
            lifecycle: true,
            list: true,
            cancel: true,
            output: true,
          },
          steering: {
            version: 1,
            transport: "request",
            upstreamTurn: "same",
            configPolicy: "active",
          },
        },
      },
      promptCapabilities: { image: true, embeddedContext: true },
      loadSession: true,
      sessionCapabilities: { resume: {}, list: {} },
    },
    authMethods: [],
  };
}

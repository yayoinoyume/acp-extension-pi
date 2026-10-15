import type { ChildProcess } from "node:child_process";
import spawn from "cross-spawn";
import { Readable, Writable } from "node:stream";
import { fileURLToPath } from "node:url";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  AgentSideConnection,
  RequestError,
  type Agent,
  type Stream,
} from "@agentclientprotocol/sdk";
import {
  LODY_EXTENSION_METHODS,
  normalizeLodyExtensionMethod,
  supportsLodySubagentEvents,
} from "acp-extension-core";
import { PiStderrDiagnostics } from "./diagnostics.js";
import { PiRpcConnection, initializeResponse } from "./connection.js";
import { PI_EXTENSIONS_ENV, parsePiLaunchArgs } from "./extensions.js";
import { resolvePiLaunch } from "./launch.js";
import { MCP_CONFIG_ENV } from "./mcp.js";
import { listPiSessions } from "./sessions.js";

const SHUTDOWN_GRACE_MS = 1_000;

function waitForExit(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null)
    return Promise.resolve();
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      child.off("exit", exited);
      resolve();
    }, SHUTDOWN_GRACE_MS);
    const exited = () => {
      clearTimeout(timeout);
      resolve();
    };
    child.once("exit", exited);
  });
}

/** One runtime per ACP connection; Pi owns its native files and tool processes. */
export function serve(stream: Stream, piArgs: string[] = []) {
  // Never forward ambient extension/resource control flags to the native process.
  const parsed = parsePiLaunchArgs(piArgs);
  let child: ChildProcess | undefined;
  let runtime: Promise<PiRpcConnection> | undefined;
  let subagentEvents = false;
  let current: PiRpcConnection | undefined;
  let cwd: string | undefined;
  let closing: Promise<void> | undefined;
  let piExitedUnexpectedly = false;
  let configDirectory: string | undefined;
  let resolveClosed!: () => void;
  const closed = new Promise<void>((resolve) => {
    resolveClosed = resolve;
  });
  const close = (): Promise<void> => {
    if (closing) return closing;
    closing = (async () => {
      current?.close();
      try {
        const owned = child;
        if (!owned?.pid) return;
        if (process.platform === "win32") owned.stdin?.end();
        else {
          // Pi's SIGTERM handler stops tracked detached tools before awaiting shutdown hooks.
          try {
            process.kill(-owned.pid, "SIGTERM");
          } catch {
            /* Already exited. */
          }
        }
        await waitForExit(owned);
        if (process.platform !== "win32") {
          try {
            process.kill(-owned.pid, "SIGKILL");
          } catch {
            /* Already exited. */
          }
          await waitForExit(owned);
        }
      } finally {
        try {
          if (configDirectory)
            rmSync(configDirectory, { recursive: true, force: true });
        } finally {
          // The executable exits, releasing its Windows Job and all descendants.
          resolveClosed();
        }
      }
    })();
    return closing;
  };
  const connection = new AgentSideConnection((client): Agent => {
    const get = async (directory?: string) => {
      if (closing) throw new Error("ACP connection closed");
      if (runtime) {
        if (directory && directory !== cwd)
          throw RequestError.invalidRequest(
            undefined,
            "Use another ACP connection for a different working directory",
          );
        return runtime;
      }
      if (!directory)
        throw RequestError.invalidRequest(
          undefined,
          "Create or resume a session first",
        );
      cwd = directory;
      configDirectory = mkdtempSync(join(tmpdir(), "lody-pi-mcp-"));
      const configPath = join(configDirectory, "servers.json");
      writeFileSync(configPath, "[]", { mode: 0o600 });
      const { command, args: launchArgs } = resolvePiLaunch([
        ...parsed.args,
        "--mode",
        "rpc",
        "--no-extensions",
        "-e",
        fileURLToPath(new URL("./extension.js", import.meta.url)),
      ]);
      child = spawn(command, launchArgs, {
        cwd,
        windowsHide: true,
        env: {
          ...process.env,
          [MCP_CONFIG_ENV]: configPath,
          [PI_EXTENSIONS_ENV]: JSON.stringify(parsed.extensions),
        },
        stdio: ["pipe", "pipe", "pipe"],
        detached: process.platform !== "win32",
      });
      child.once("exit", () => {
        if (!closing) {
          piExitedUnexpectedly = true;
          process.exitCode = 1;
        }
        void close();
      });
      child.once("error", (error) => {
        console.error("[acp-extension-pi] Failed to launch Pi:", error);
        process.exitCode = 1;
        void close();
      });
      const diagnostics = new PiStderrDiagnostics(
        () => !closing || piExitedUnexpectedly,
      );
      child.stderr!.setEncoding("utf8");
      child.stderr!.pipe(process.stderr, { end: false });
      child.stderr!.on("data", (chunk) => diagnostics.append(chunk));
      const pi = new PiRpcConnection(
        {
          writable: Writable.toWeb(child.stdin!),
          readable: Readable.toWeb(child.stdout!),
        },
        {
          subagentEvents,
          configureMcp: async (servers) => {
            // Configuration exclusion keeps this write before the next runtime loads it.
            if (closing) throw new Error("ACP connection closed");
            writeFileSync(configPath, JSON.stringify(servers), {
              mode: 0o600,
            });
          },
          update: (notification) => client.sessionUpdate(notification),
          // Notifications are sent in Pi event order. Lody owns the application lease
          // and gates session updates after the matching Core notification arrives.
          extension: (method, params) => client.extNotification(method, params),
          usage: (usage) => {
            void client
              .extNotification(LODY_EXTENSION_METHODS.sessionUsageUpdate, {
                ...usage,
              })
              .catch(close);
          },
          question: (request) => client.unstable_createElicitation(request),
        },
        (error) => diagnostics.decorateError(error),
        // Lets initialize probe the runtime for its dispatch capabilities.
        { command, entry: launchArgs[0] ?? command },
      );
      current = pi;
      runtime = Promise.race([
        pi.initialize({ protocolVersion: 1 }).then(() => {
          diagnostics.stop();
          return pi;
        }),
        new Promise<never>((_resolve, reject) => child!.once("error", reject)),
      ]);
      return runtime;
    };
    const validate = (request: { mcpServers?: unknown[] }) => {
      if (
        request.mcpServers?.some(
          (server) =>
            !server || typeof server !== "object" || !("command" in server),
        )
      )
        throw RequestError.invalidRequest(
          undefined,
          "Pi supports stdio MCP servers only",
        );
    };
    return {
      initialize: async (request) => {
        subagentEvents = supportsLodySubagentEvents(request.clientCapabilities);
        return initializeResponse();
      },
      authenticate: async () => {
        throw RequestError.invalidRequest(
          undefined,
          "Authenticate through Pi on the execution machine",
        );
      },
      newSession: async (request) => {
        validate(request);
        return (await get(request.cwd)).newSession(request);
      },
      resumeSession: async (request) => {
        validate(request);
        return (await get(request.cwd)).resumeSession(request);
      },
      loadSession: async (request) => {
        validate(request);
        return (await get(request.cwd)).loadSession(request);
      },
      listSessions: listPiSessions,
      prompt: async (request) => (await get()).prompt(request),
      cancel: async (request) => {
        await (await get()).cancel(request);
      },
      setSessionConfigOption: async (request) =>
        (await get()).setSessionConfigOption(request),
      extMethod: async (method, params) =>
        (await get()).request(normalizeLodyExtensionMethod(method), params),
    };
  }, stream);
  connection.signal.addEventListener("abort", () => void close(), {
    once: true,
  });
  return { connection, close, closed };
}

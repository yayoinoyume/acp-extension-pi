import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";

/** Pi reports how an accepted prompt was dispatched from 0.99.0 onward. */
const DISPOSITION_INTRODUCED_IN = "0.99.0";
const PI_PACKAGE_NAME = "@earendil-works/pi-coding-agent";
const PROBE_TIMEOUT_MS = 5000;

export type PiCapabilities = {
  /** Pi's prompt response carries a disposition of handled/queued/started. */
  hasDisposition: boolean;
  /** Resolved Pi version, when detection succeeded. */
  version?: string;
};

/**
 * Ask the runtime to print its semver. The flag exits before any session,
 * network or extension work, so probing cannot disturb the session.
 */
export function probeVersionFromLaunch(
  command: string,
  entry: string,
): Promise<string | undefined> {
  return new Promise((resolve) => {
    let stdout = "";
    let settled = false;
    let child: ReturnType<typeof spawn> | undefined;
    const done = (value?: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => {
      child?.kill();
      done(undefined);
    }, PROBE_TIMEOUT_MS);
    try {
      child = spawn(command, [entry, "--version"], {
        stdio: ["ignore", "pipe", "ignore"],
      });
    } catch {
      done(undefined);
      return;
    }
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.once("error", () => done(undefined));
    child.once("close", () => done(parseVersion(stdout)));
  });
}

/** A package.json next to the launch entry, when it is Pi's own. */
export function versionFromPackageJson(entry: string): string | undefined {
  try {
    const manifest = JSON.parse(
      readFileSync(join(dirname(entry), "..", "..", "package.json"), "utf8"),
    ) as { name?: string; version?: string };
    // A resolve against an unrelated package yields that package's manifest.
    if (manifest.name !== PI_PACKAGE_NAME) return undefined;
    return parseVersion(manifest.version ?? "");
  } catch {
    return undefined;
  }
}

export type CapabilityProbes = {
  launch: (command: string, entry: string) => Promise<string | undefined>;
  manifest: (entry: string) => string | undefined;
  steer: (fields: Record<string, unknown>) => Promise<unknown>;
};

export const liveProbes: CapabilityProbes = {
  launch: probeVersionFromLaunch,
  manifest: versionFromPackageJson,
  steer: async (fields) => {
    throw new Error("no runtime to probe");
  },
};

/** Cheapest probe first; every failure degrades to the legacy path. */
export async function detectCapabilities(
  input: { command: string; entry: string },
  probes: CapabilityProbes = liveProbes,
): Promise<PiCapabilities> {
  const launch = await probes.launch(input.command, input.entry);
  if (launch) return fromVersion(launch);
  const manifest = probes.manifest(input.entry);
  if (manifest) return fromVersion(manifest);
  return { hasDisposition: await dispositionFromSteer(probes.steer) };
}

async function dispositionFromSteer(
  steer: (fields: Record<string, unknown>) => Promise<unknown>,
): Promise<boolean> {
  try {
    // The transport resolves with the response payload, not its envelope.
    const payload = (await steer({ message: "" })) as
      { disposition?: string } | undefined;
    return typeof payload?.disposition === "string";
  } catch {
    return false;
  }
}

function fromVersion(version: string): PiCapabilities {
  return {
    hasDisposition: supportsDisposition(version),
    version,
  };
}

function supportsDisposition(version: string): boolean {
  return compareVersions(version, DISPOSITION_INTRODUCED_IN) >= 0;
}

/** Numeric dot-part comparison, enough for a released Pi version pair. */
function compareVersions(left: string, right: string): number {
  const a = left.split(".").map((part) => Number.parseInt(part, 10) || 0);
  const b = right.split(".").map((part) => Number.parseInt(part, 10) || 0);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    if (diff) return diff;
  }
  return 0;
}

function parseVersion(raw: string): string | undefined {
  const match = /(\d+\.\d+\.\d+(?:-[\w.]+)?)/.exec(raw.trim());
  return match ? match[1] : undefined;
}

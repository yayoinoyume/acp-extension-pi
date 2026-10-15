import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  detectCapabilities,
  probeVersionFromLaunch,
  versionFromPackageJson,
} from "../src/capability.js";

const node = process.execPath;

describe("probeVersionFromLaunch", () => {
  it("reads the semver a real entry prints for --version", async () => {
    const directory = mkdtempSync(join(tmpdir(), "pi-probe-"));
    const entry = join(directory, "cli.js");
    writeFileSync(entry, 'process.stdout.write("1.2.3\\n");\n');
    await expect(probeVersionFromLaunch(node, entry)).resolves.toBe("1.2.3");
  });

  it("tolerates chatter around the version", async () => {
    const directory = mkdtempSync(join(tmpdir(), "pi-probe-"));
    const entry = join(directory, "cli.js");
    writeFileSync(entry, 'process.stdout.write("pi v0.99.1 (node)\\n");\n');
    await expect(probeVersionFromLaunch(node, entry)).resolves.toBe("0.99.1");
  });

  it("resolves undefined when the entry prints no version", async () => {
    const directory = mkdtempSync(join(tmpdir(), "pi-probe-"));
    const entry = join(directory, "cli.js");
    writeFileSync(entry, 'process.stdout.write("no version here\\n");\n');
    await expect(probeVersionFromLaunch(node, entry)).resolves.toBeUndefined();
  });

  it("resolves undefined when the entry does not exist", async () => {
    await expect(
      probeVersionFromLaunch(node, "/nonexistent/cli.js"),
    ).resolves.toBeUndefined();
  });
});

describe("versionFromPackageJson", () => {
  it("reads a Pi manifest beside the entry", () => {
    const directory = mkdtempSync(join(tmpdir(), "pi-pkg-"));
    const entry = join(directory, "pkg", "dist", "bundle", "cli.js");
    mkdirSync(join(directory, "pkg", "dist", "bundle"), { recursive: true });
    writeFileSync(entry, "");
    writeFileSync(
      join(directory, "pkg", "package.json"),
      JSON.stringify({
        name: "@earendil-works/pi-coding-agent",
        version: "1.1.0",
      }),
    );
    expect(versionFromPackageJson(entry)).toBe("1.1.0");
  });

  it("ignores a manifest belonging to another package", () => {
    const directory = mkdtempSync(join(tmpdir(), "pi-pkg-"));
    const entry = join(directory, "pkg", "dist", "bundle", "cli.js");
    mkdirSync(join(directory, "pkg", "dist", "bundle"), { recursive: true });
    writeFileSync(entry, "");
    writeFileSync(
      join(directory, "pkg", "package.json"),
      JSON.stringify({ name: "acp-extension-pi", version: "0.2.0" }),
    );
    expect(versionFromPackageJson(entry)).toBeUndefined();
  });
});

describe("detectCapabilities", () => {
  const input = { command: node, entry: "/pkg/bundle/cli.js" };
  const deadSteer = async () => {
    throw new Error("no runtime to probe");
  };

  it("disables disposition for 0.87.x runtimes", async () => {
    const capabilities = await detectCapabilities(input, {
      launch: async () => "0.87.0",
      manifest: () => undefined,
      steer: deadSteer,
    });
    expect(capabilities).toEqual({ hasDisposition: false, version: "0.87.0" });
  });

  it("enables disposition from 0.99.0 onward", async () => {
    const capabilities = await detectCapabilities(input, {
      launch: async () => "0.99.1",
      manifest: () => undefined,
      steer: deadSteer,
    });
    expect(capabilities).toEqual({ hasDisposition: true, version: "0.99.1" });
  });

  it("reads a 1.x runtime as disposition-capable", async () => {
    const capabilities = await detectCapabilities(input, {
      launch: async () => "1.2.0",
      manifest: () => undefined,
      steer: deadSteer,
    });
    expect(capabilities).toEqual({ hasDisposition: true, version: "1.2.0" });
  });

  it("falls back to the manifest when the probe fails", async () => {
    const capabilities = await detectCapabilities(input, {
      launch: async () => undefined,
      manifest: () => "1.1.0",
      steer: deadSteer,
    });
    expect(capabilities).toEqual({ hasDisposition: true, version: "1.1.0" });
  });

  it("asks the runtime itself when no version can be read", async () => {
    const seen: Record<string, unknown>[] = [];
    const capabilities = await detectCapabilities(input, {
      launch: async () => undefined,
      manifest: () => undefined,
      steer: async (fields) => {
        seen.push(fields);
        return { disposition: "handled" };
      },
    });
    expect(capabilities).toEqual({ hasDisposition: true });
    // An empty steer queues nothing and starts no model turn.
    expect(seen).toEqual([{ message: "" }]);
  });

  it("treats a failed steer probe as no disposition", async () => {
    const capabilities = await detectCapabilities(input, {
      launch: async () => undefined,
      manifest: () => undefined,
      steer: deadSteer,
    });
    expect(capabilities).toEqual({ hasDisposition: false });
  });

  it("ignores a steer response without a disposition", async () => {
    const capabilities = await detectCapabilities(input, {
      launch: async () => undefined,
      manifest: () => undefined,
      steer: async () => ({}),
    });
    expect(capabilities).toEqual({ hasDisposition: false });
  });
});

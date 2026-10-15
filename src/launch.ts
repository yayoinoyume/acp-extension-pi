import { fileURLToPath } from "node:url";

export const PI_PATH_ENV = "LODY_PI_PATH";

export type PiLaunch = {
  command: string;
  args: string[];
  /** The Pi entry the command runs; with --version it reports the semver. */
  entry: string;
};

/**
 * Resolve the Pi CLI to launch. LODY_PI_PATH overrides the packaged CLI; the
 * result is handed straight to cross-spawn.
 */
export function resolvePiLaunch(
  extraArgs: string[],
  env: NodeJS.ProcessEnv = process.env,
  resolveBundledEntry: () => string = defaultBundledEntry,
): PiLaunch {
  const override = env[PI_PATH_ENV]?.trim();
  if (!override) {
    const entry = resolveBundledEntry();
    return { command: process.execPath, args: [entry, ...extraArgs], entry };
  }
  return { command: override, args: extraArgs, entry: override };
}

function defaultBundledEntry(): string {
  return fileURLToPath(
    new URL(
      "./bundle/cli.js",
      import.meta.resolve("@earendil-works/pi-coding-agent"),
    ),
  );
}

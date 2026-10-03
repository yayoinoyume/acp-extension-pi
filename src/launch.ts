import { fileURLToPath } from "node:url";

export const PI_PATH_ENV = "LODY_PI_PATH";

export type PiLaunch = {
  command: string;
  args: string[];
  shell?: boolean;
};

/**
 * Resolve the Pi CLI to launch. LODY_PI_PATH points at a user-supplied Pi
 * executable; without it the packaged official Pi CLI is used. The caller owns
 * appending mode/extension flags to the returned args.
 */
export function resolvePiLaunch(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  resolveBundledEntry: () => string = defaultBundledEntry,
): PiLaunch {
  const override = env[PI_PATH_ENV]?.trim();
  if (override) {
    // Mirrors the CODEX_PATH contract in acp-extension-codex: spawn the user
    // binary verbatim; Windows needs a shell because the path may be a .cmd shim.
    return platform === "win32"
      ? { command: `"${override}"`, args: [], shell: true }
      : { command: override, args: [] };
  }
  return { command: process.execPath, args: [resolveBundledEntry()] };
}

function defaultBundledEntry(): string {
  return fileURLToPath(
    new URL(
      "./bundle/cli.js",
      import.meta.resolve("@earendil-works/pi-coding-agent"),
    ),
  );
}

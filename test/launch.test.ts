import { describe, expect, it } from "vitest";
import { PI_PATH_ENV, resolvePiLaunch } from "../src/launch.js";

describe("resolvePiLaunch", () => {
  it("falls back to the packaged Pi CLI run by the current Node", () => {
    const launch = resolvePiLaunch({}, "linux", () => "/pkg/bundle/cli.js");
    expect(launch.command).toBe(process.execPath);
    expect(launch.args).toEqual(["/pkg/bundle/cli.js"]);
    expect(launch.shell).toBeUndefined();
  });

  it("spawns a user-supplied POSIX executable verbatim", () => {
    const launch = resolvePiLaunch(
      { [PI_PATH_ENV]: "/usr/local/bin/pi" },
      "linux",
    );
    expect(launch).toEqual({ command: "/usr/local/bin/pi", args: [] });
  });

  it("ignores blank overrides", () => {
    const launch = resolvePiLaunch(
      { [PI_PATH_ENV]: "   " },
      "linux",
      () => "/pkg/bundle/cli.js",
    );
    expect(launch.args[0]).toBe("/pkg/bundle/cli.js");
  });

  it("quotes user-supplied Windows shims behind a shell", () => {
    const launch = resolvePiLaunch(
      { [PI_PATH_ENV]: "C:\\Users\\me\\AppData\\pi.cmd" },
      "win32",
    );
    expect(launch.command).toBe('"C:\\Users\\me\\AppData\\pi.cmd"');
    expect(launch.shell).toBe(true);
  });
});

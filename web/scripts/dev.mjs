// SPDX-License-Identifier: MPL-2.0

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, delimiter, resolve } from "node:path";

const webDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectDirectory = resolve(webDirectory, "..");
const apiPort = Number.parseInt(process.env.TSQ_API_PORT ?? "8765", 10);
const webPort = Number.parseInt(process.env.TSQ_WEB_PORT ?? "3000", 10);

if (
  !Number.isInteger(apiPort) || apiPort < 1 || apiPort > 65535 ||
  !Number.isInteger(webPort) || webPort < 1 || webPort > 65535
) {
  console.error("TSQ_API_PORT and TSQ_WEB_PORT must be integers from 1 to 65535.");
  process.exit(2);
}

const apiOrigin = `http://127.0.0.1:${apiPort}`;
const processes = new Set();
let stopping = false;

function launch(name, command, args, options) {
  const child = spawn(command, args, {
    ...options,
    detached: process.platform !== "win32",
    stdio: "inherit",
  });
  child.tsqName = name;
  processes.add(child);
  child.once("exit", () => processes.delete(child));
  return child;
}

function terminate(child, signal) {
  if (child.exitCode !== null || child.signalCode !== null || child.pid === undefined) {
    return;
  }
  try {
    if (process.platform !== "win32") {
      process.kill(-child.pid, signal);
    } else {
      child.kill(signal);
    }
  } catch (error) {
    if (error?.code !== "ESRCH") {
      console.error(`Could not stop ${child.tsqName}:`, error);
    }
  }
}

function waitForExit(child) {
  return new Promise((resolveExit) => {
    child.once("error", (error) =>
      resolveExit({ name: child.tsqName, code: 1, signal: null, error }),
    );
    child.once("exit", (code, signal) =>
      resolveExit({ name: child.tsqName, code, signal, error: null }),
    );
  });
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function waitForApi(apiExit) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const probe = fetch(`${apiOrigin}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(1_000),
    })
      .then((response) => (response.ok ? "ready" : "retry"))
      .catch(() => "retry");
    const outcome = await Promise.race([
      probe,
      apiExit.then((result) => ({ exited: result })),
    ]);
    if (outcome === "ready") {
      return;
    }
    if (typeof outcome === "object" && outcome !== null && "exited" in outcome) {
      const result = outcome.exited;
      throw result.error ?? new Error(
        `The TSQ API exited before it became ready (${result.code ?? result.signal ?? "unknown"}).`,
      );
    }
    await delay(150);
  }
  throw new Error(`The TSQ API did not become ready at ${apiOrigin} within 30 seconds.`);
}

async function stopAll() {
  if (stopping) {
    return;
  }
  stopping = true;
  for (const child of processes) {
    terminate(child, "SIGTERM");
  }
  await delay(1_500);
  for (const child of processes) {
    terminate(child, "SIGKILL");
  }
}

async function main() {
  let resolveSignal;
  const interrupted = new Promise((resolveInterrupted) => {
    resolveSignal = resolveInterrupted;
  });
  process.once("SIGINT", () => resolveSignal({ name: "signal", signal: "SIGINT" }));
  process.once("SIGTERM", () => resolveSignal({ name: "signal", signal: "SIGTERM" }));

  console.log("Starting the TSQ engine and interface…");
  const api = launch(
    "TSQ API",
    resolve(projectDirectory, "serve"),
    [
      "--port", String(apiPort),
      "--allow-origin", `http://localhost:${webPort}`,
      "--allow-origin", `http://127.0.0.1:${webPort}`,
    ],
    {
      cwd: projectDirectory,
      env: {
        ...process.env,
        PYTHONPATH: [
          resolve(projectDirectory, "src"),
          process.env.PYTHONPATH,
        ].filter(Boolean).join(delimiter),
      },
    },
  );
  const apiExit = waitForExit(api);
  const readiness = await Promise.race([
    waitForApi(apiExit).then(() => null),
    interrupted,
  ]);
  if (readiness !== null) {
    await stopAll();
    process.exitCode = 0;
    return;
  }

  const web = launch(
    "TSQ web",
    process.execPath,
    [
      resolve(webDirectory, "node_modules", "vinext", "dist", "cli.js"),
      "dev",
      "--port",
      String(webPort),
    ],
    {
      cwd: webDirectory,
      env: {
        ...process.env,
        TSQ_API_ORIGIN: apiOrigin,
        TSQ_WEB_PORT: String(webPort),
        WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
      },
    },
  );
  const webExit = waitForExit(web);

  const result = await Promise.race([apiExit, webExit, interrupted]);
  if (result.name !== "signal" && !stopping) {
    const detail = result.error?.message ?? result.code ?? result.signal ?? "unknown";
    console.error(`${result.name} stopped unexpectedly (${detail}).`);
  }
  await stopAll();
  process.exitCode = result.name === "signal" ? 0 : (result.code ?? 1);
}

main().catch(async (error) => {
  console.error(error instanceof Error ? error.message : error);
  await stopAll();
  process.exitCode = 1;
});

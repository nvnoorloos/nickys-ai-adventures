import fs from "node:fs";
import path from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { runPluginCommandWithTimeout } from "openclaw/plugin-sdk";

/**
 * Where the OpenClaw binary lives on this machine.
 *
 * - We allow override via OPENCLAW_BIN for flexibility across environments.
 * - The fallback keeps local development simple.
 */
const OPENCLAW_BIN = process.env.OPENCLAW_BIN?.trim() || "/usr/local/bin/openclaw";

/**
 * Root directory that contains developer projects targetable by /dev.
 *
 * Security model:
 * - /dev only accepts direct child folders of this root.
 * - No nested paths, traversal, or absolute path injection.
 */
const PROJECTS_ROOT = "/workspace/projects";

/**
 * Time budget for each agent turn (code-bot and review-bot).
 *
 * We intentionally keep these equal and predictable.
 * The gateway timeout is passed as string because the CLI flag expects a string value.
 */
const AGENT_TIMEOUT_MS = 10 * 60 * 1000;
const GATEWAY_TIMEOUT_MS = String(AGENT_TIMEOUT_MS);

type GatewayAgentPayload = {
  text?: string;
  mediaUrl?: string | null;
  mediaUrls?: string[];
};

type GatewayAgentResponse = {
  runId?: string;
  status?: string;
  summary?: string;
  result?: {
    payloads?: GatewayAgentPayload[];
    meta?: unknown;
  };
};

type ProjectResolution = {
  projectKey: string;
  repoPath: string;
  rootPath: string;
};

type ParsedDevArgs = {
  projectFolder: string;
  task: string;
};

type ProgressTarget = {
  channel: string;
  to?: string;
  accountId?: string;
  messageThreadId?: number;
};

/**
 * Parse `/dev` arguments in the strict format:
 *
 *   <project-folder> - <task prompt>
 *
 * Why strict?
 * - avoids ambiguous parsing in chat UIs
 * - produces deterministic usage errors
 * - protects project resolution from path tricks
 */
function parseDevArgs(rawArgs: string): ParsedDevArgs {
  const args = rawArgs.trim();
  const separatorIndex = args.indexOf(" - ");
  if (separatorIndex === -1) {
    throw new Error(
      "Usage: /dev <project-folder> - <task prompt>\n\nExample:\n/dev my-app - add a health endpoint",
    );
  }

  const projectFolder = args.slice(0, separatorIndex).trim();
  const task = args.slice(separatorIndex + 3).trim();

  if (!projectFolder || !task) {
    throw new Error(
      "Usage: /dev <project-folder> - <task prompt>\n\nExample:\n/dev my-app - add a health endpoint",
    );
  }

  // Guardrail: only direct child folders under PROJECTS_ROOT are allowed.
  if (projectFolder.includes("/") || projectFolder.includes("\\")) {
    throw new Error("Project folder must be a direct child of /workspace/projects.");
  }

  if (projectFolder === "." || projectFolder === "..") {
    throw new Error("Project folder must be a direct child of /workspace/projects.");
  }

  return { projectFolder, task };
}

/**
 * Resolve and validate the target project folder.
 *
 * Security checks in order:
 * 1) Ensure PROJECTS_ROOT is accessible.
 * 2) Resolve target path from known root + user-provided folder name.
 * 3) Ensure resolved path still sits directly under PROJECTS_ROOT.
 * 4) Ensure resolved path exists and is a directory.
 */
function resolveProject(projectFolder: string): ProjectResolution {
  fs.accessSync(PROJECTS_ROOT, fs.constants.R_OK | fs.constants.W_OK);
  const repoPath = path.resolve(PROJECTS_ROOT, projectFolder);

  if (path.dirname(repoPath) !== PROJECTS_ROOT) {
    throw new Error("Project folder must stay inside /workspace/projects.");
  }

  const stat = fs.statSync(repoPath, { throwIfNoEntry: false });
  if (!stat?.isDirectory()) {
    throw new Error(
      `Project folder not found: ${repoPath}\n\nCreate it first inside /workspace/projects, then retry.`,
    );
  }

  return {
    projectKey: projectFolder,
    repoPath,
    rootPath: PROJECTS_ROOT,
  };
}

/**
 * Parse JSON safely. Returns null instead of throwing.
 *
 * This keeps downstream logic simple:
 * - success path reads typed object
 * - failure path emits explicit plugin error
 */
function parseJsonObject<T>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/**
 * Extract human-usable text from a gateway agent response.
 *
 * Preferred source: payload text chunks.
 * Fallback source: summary field.
 */
function extractAgentText(response: GatewayAgentResponse): string {
  const payloads = Array.isArray(response.result?.payloads) ? response.result?.payloads : [];
  const texts = payloads
    .map((payload) => (typeof payload.text === "string" ? payload.text.trim() : ""))
    .filter(Boolean);

  if (texts.length > 0) {
    return texts.join("\n\n");
  }

  return typeof response.summary === "string" ? response.summary.trim() : "";
}

/**
 * Build a strict instruction set for code-bot.
 *
 * We ask for fixed sections so review-bot and humans can scan output quickly.
 */
function buildCodeBotPrompt(task: string, project: ProjectResolution): string {
  return [
    `Work in the project directory \`${project.repoPath}\`.`,
    "Use the existing files in that directory if present.",
    "Implement the request directly in that directory.",
    "Keep changes scoped to that project directory unless the request explicitly requires otherwise.",
    "When you finish, return exactly these sections in plain text:",
    "Status",
    "<done or blocked>",
    "",
    "Project",
    `<${project.projectKey}>`,
    "",
    "Files",
    "<one path per line>",
    "",
    "Summary",
    "<short implementation summary>",
    "",
    "Notes",
    "<important caveats or 'none'>",
    "",
    "Request",
    task.trim(),
  ].join("\n");
}

/**
 * Build the review-bot prompt using original task + code-bot output.
 *
 * Review-bot acts as a validator, not an implementer.
 */
function buildReviewBotPrompt(
  task: string,
  project: ProjectResolution,
  codeBotOutput: string,
): string {
  return [
    `Review the code-bot output for work in \`${project.repoPath}\`.`,
    "You are the reviewer only.",
    "Decide whether the implementation satisfies the request and call out concrete issues.",
    "Assume the code-bot already made the changes. Base your review on the reported files and summary.",
    "Return exactly these sections in plain text:",
    "Verdict",
    "<approved or rejected>",
    "",
    "Summary",
    "<one short paragraph>",
    "",
    "Issues",
    "<bullet list or 'none'>",
    "",
    "Original Request",
    task.trim(),
    "",
    "Code Bot Output",
    codeBotOutput.trim(),
  ].join("\n");
}

/**
 * Assemble final user-facing response.
 *
 * The final shape is intentionally chat-friendly:
 * - quick header
 * - project context
 * - code-bot section
 * - review-bot section
 */
function buildFinalReply(params: {
  project: ProjectResolution;
  codeBotOutput: string;
  reviewBotOutput: string;
}): string {
  return [
    `🎉 /dev — done`,
    ``,
    `📁 Project: ${params.project.projectKey}`,
    `📍 Location: ${params.project.repoPath}`,
    `🗂️ Project Root: ${params.project.rootPath}`,
    ``,
    `👨‍💻 Code Bot`,
    params.codeBotOutput.trim() || "(no output)",
    ``,
    `🕵️‍♂️ Review Bot`,
    params.reviewBotOutput.trim() || "(no output)",
  ].join("\n");
}

/**
 * Create stable session keys per agent + project.
 *
 * Benefit: each project keeps a useful rolling context for that specific bot role.
 */
function buildAgentSessionKey(agentId: string, projectKey: string): string {
  return `agent:${agentId}:dev:${projectKey}`;
}

/**
 * Run an OpenClaw agent turn via gateway CLI.
 *
 * Notable choices:
 * - deliver:false so this plugin controls user-visible output
 * - idempotencyKey to reduce accidental duplicate execution
 * - expect-final/json to guarantee parsable output contract
 */
async function runAgent(params: {
  agentId: string;
  sessionKey: string;
  message: string;
}): Promise<GatewayAgentResponse> {
  const result = await runPluginCommandWithTimeout({
    argv: [
      OPENCLAW_BIN,
      "gateway",
      "call",
      "agent",
      "--params",
      JSON.stringify({
        message: params.message,
        agentId: params.agentId,
        sessionKey: params.sessionKey,
        deliver: false,
        idempotencyKey: `dev-${params.agentId}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        label: `dev-${params.agentId}`,
        timeout: Math.floor(AGENT_TIMEOUT_MS / 1000),
      }),
      "--expect-final",
      "--timeout",
      GATEWAY_TIMEOUT_MS,
      "--json",
    ],
    timeoutMs: AGENT_TIMEOUT_MS + 30_000,
  });

  if (result.code !== 0) {
    throw new Error(result.stderr.trim() || result.stdout.trim() || `${params.agentId} run failed`);
  }

  const parsed = parseJsonObject<GatewayAgentResponse>(result.stdout);
  if (!parsed) {
    throw new Error(`${params.agentId} returned invalid JSON`);
  }

  return parsed;
}

/**
 * Send a progress message to the same chat/thread where /dev was called.
 *
 * These updates are best-effort. Failures are logged but do not block core flow.
 */
async function sendProgressUpdate(target: ProgressTarget, text: string): Promise<void> {
  if (!target.to) {
    return;
  }

  const argv = [
    OPENCLAW_BIN,
    "message",
    "send",
    "--channel",
    target.channel,
    "--target",
    target.to,
    "--message",
    text,
  ];

  if (target.accountId) {
    argv.push("--account", target.accountId);
  }

  if (target.messageThreadId != null) {
    argv.push("--thread-id", String(target.messageThreadId));
  }

  const result = await runPluginCommandWithTimeout({
    argv,
    timeoutMs: 30_000,
  });

  if (result.code !== 0) {
    throw new Error(result.stderr.trim() || result.stdout.trim() || "progress update failed");
  }
}

/**
 * Register a /dev command variant.
 *
 * We call this twice:
 * - "dev"
 * - "dev@<botname>"
 *
 * Telegram groups frequently include the bot suffix, so dual registration prevents
 * command-matching surprises.
 */
function registerDevCommand(api: OpenClawPluginApi, name: string) {
  api.registerCommand({
    name,
    description: "Run code-bot and then review-bot for a development task.",
    acceptsArgs: true,
    handler: async (ctx) => {

      const rawArgs = ctx.args?.trim() ?? "";
      if (!rawArgs) {
        return {
          text:
            `Usage: /${name} <project-folder> - <task prompt>\n\n` +
            "Example:\n" +
            "/dev my-app - add a hello world endpoint",
        };
      }

      let parsedArgs: ParsedDevArgs;
      try {
        parsedArgs = parseDevArgs(rawArgs);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { text: message };
      }

      let project: ProjectResolution;
      try {
        project = resolveProject(parsedArgs.projectFolder);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        api.logger.error(`dev-command: project resolution failed: ${message}`);
        return { text: `Failed to resolve project.\n\n${message}` };
      }

      const progressTarget: ProgressTarget = {
        channel: ctx.channel,
        to: ctx.to,
        accountId: ctx.accountId,
        messageThreadId: ctx.messageThreadId,
      };

      try {
        // Stage 1: implementation pass.
        await sendProgressUpdate(
          progressTarget,
          `🛠️ /dev — started\n\n📁 Project: \`${project.projectKey}\`\n👨‍💻 Step 1/2: code-bot is busy...`,
        ).catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          api.logger.warn(`dev-command: failed to send code-bot-start update: ${message}`);
        });

        const codeBotResponse = await runAgent({
          agentId: "code-bot",
          sessionKey: buildAgentSessionKey("code-bot", project.projectKey),
          message: buildCodeBotPrompt(parsedArgs.task, project),
        });
        const codeBotOutput = extractAgentText(codeBotResponse);
        if (!codeBotOutput) {
          throw new Error("code-bot returned no usable output");
        }

        // Stage 2: review pass.
        await sendProgressUpdate(
          progressTarget,
          `✅ code-bot ready\n\n📁 Project: \`${project.projectKey}\`\n🕵️‍♂️ Step 2/2: review-bot is watching...`,
        ).catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          api.logger.warn(`dev-command: failed to send review-bot-start update: ${message}`);
        });

        const reviewBotResponse = await runAgent({
          agentId: "review-bot",
          sessionKey: buildAgentSessionKey("review-bot", project.projectKey),
          message: buildReviewBotPrompt(parsedArgs.task, project, codeBotOutput),
        });
        const reviewBotOutput = extractAgentText(reviewBotResponse);
        if (!reviewBotOutput) {
          throw new Error("review-bot returned no usable output");
        }

        await sendProgressUpdate(
          progressTarget,
          `✅ review-bot ready\n\n📦 Result is being assembled… (last step)`,
        ).catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          api.logger.warn(`dev-command: failed to send review-bot-finished update: ${message}`);
        });

        return {
          text: buildFinalReply({
            project,
            codeBotOutput,
            reviewBotOutput,
          }),
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        api.logger.error(`dev-command: /dev flow failed: ${message}`);

        await sendProgressUpdate(
          progressTarget,
          `🧨 /dev — failed\n\n📁 Project: \`${project.projectKey}\`\n❌ Error: ${message}`,
        ).catch((sendError) => {
          const sendMessage = sendError instanceof Error ? sendError.message : String(sendError);
          api.logger.warn(`dev-command: failed to send error update: ${sendMessage}`);
        });

        return {
          text:
            `🧨 /dev — failed\n\n` +
            `📁 Project: ${project.projectKey}\n` +
            `📍 Location: ${project.repoPath}\n` +
            `🗂️ Project Root: ${project.rootPath}\n\n` +
            `❌ Error\n${message}`,
        };
      }
    },
  });
}

/**
 * Plugin entrypoint.
 *
 * Telegram can send commands with bot suffixes in groups (`/dev@botname`).
 * Register both forms to make the command reliable in 1:1 and group chats.
 */
export default function register(api: OpenClawPluginApi) {
  registerDevCommand(api, "dev");
  registerDevCommand(api, "dev@your_bot_username");
}

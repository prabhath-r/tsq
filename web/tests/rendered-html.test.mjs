import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { inflateSync } from "node:zlib";
import test from "node:test";

const identityPattern = /\b(?:openai(?:_codex)?|chatgpt|codex|gpt[-_\s]?\d(?:\.\d+)?)\b|provider_name|model_name/i;
const publicCopyIdentityPattern = /\b(?:openai(?:_codex)?|chatgpt|codex|gpt[-_\s]?\d(?:\.\d+)?|provider)\b|provider_name|model_name/i;
const privatePathPattern = /\/Users\/|[A-Za-z]:\\Users\\|(?:^|[/\\'"\s])\.env(?:[.\w-]*)(?:$|[/\\'"\s])|corpus[/\\]topics|retrieval_augmented_generation\.json|file:\/\//im;
const rawBankShapePattern = /["'](?:question_id|family_id|learning_objective_id|answer_key|correct_answer|canonical_answer)["']\s*:|["']correct["']\s*:\s*(?:true|false)/i;
const imageMetadataPattern = /OpenAI Media|OpenAI OpCo|Codex|trainedAlgorithmicMedia|igpt-image|openai_codex|provider_name|model_name/i;

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

async function readPublicSource() {
  const files = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/data.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);

  return files.join("\n");
}

async function readBuiltClientText() {
  const assetsUrl = new URL("../dist/client/assets/", import.meta.url);
  const assetNames = await readdir(assetsUrl);
  const textAssetNames = assetNames.filter((name) => /\.(?:css|html|js|json)$/i.test(name));
  const assets = await Promise.all(
    textAssetNames.map((name) => readFile(new URL(name, assetsUrl), "utf8")),
  );

  return assets.join("\n");
}

async function readReleasedQuestions() {
  const corpusUrl = new URL("../../corpus/topics/", import.meta.url);
  const topicNames = (await readdir(corpusUrl)).filter((name) => name.endsWith(".json"));
  const topics = await Promise.all(
    topicNames.map(async (name) => JSON.parse(await readFile(new URL(name, corpusUrl), "utf8"))),
  );

  return topics.flatMap((topic) => topic.questions ?? []);
}

function pngMetadataText(buffer) {
  const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  assert.deepEqual(buffer.subarray(0, pngSignature.length), pngSignature, "asset must be a PNG");

  const metadata = [buffer.toString("latin1")];
  let offset = pngSignature.length;

  while (offset + 12 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > buffer.length) break;
    const chunk = buffer.subarray(dataStart, dataEnd);

    if (type === "tEXt" || type === "eXIf") {
      metadata.push(chunk.toString("utf8"));
    } else if (type === "zTXt") {
      const separator = chunk.indexOf(0);
      if (separator >= 0 && separator + 2 <= chunk.length) {
        metadata.push(chunk.subarray(0, separator).toString("utf8"));
        metadata.push(inflateSync(chunk.subarray(separator + 2)).toString("utf8"));
      }
    } else if (type === "iTXt") {
      const keywordEnd = chunk.indexOf(0);
      if (keywordEnd >= 0 && keywordEnd + 3 <= chunk.length) {
        metadata.push(chunk.subarray(0, keywordEnd).toString("utf8"));
        const compressed = chunk[keywordEnd + 1] === 1;
        let textStart = keywordEnd + 3;
        for (let separators = 0; separators < 2 && textStart < chunk.length; separators += 1) {
          const nextSeparator = chunk.indexOf(0, textStart);
          textStart = nextSeparator < 0 ? chunk.length : nextSeparator + 1;
        }
        const text = chunk.subarray(textStart);
        metadata.push((compressed ? inflateSync(text) : text).toString("utf8"));
      }
    }

    offset = dataEnd + 4;
    if (type === "IEND") break;
  }

  return metadata.join("\n");
}

test("server-renders the TSQ product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>TSQ — The Second Question<\/title>/i);
  assert.match(html, /The Second Question/);
  assert.match(html, /Opening the shared TSQ database and exact corpus/);
  assert.match(html, /same release, learner projections, pending decisions, and event ledger used by the CLI/);
  assert.match(html, /No sessions yet/);
  assert.match(html, /aria-label="Primary navigation"/);
  assert.doesNotMatch(html, /aria-label="Answer choices"/);
  assert.doesNotMatch(html, /Example learner|job_048|claim_003|43 selected responses/i);
  assert.doesNotMatch(html, /\/og\.png/i);
  assert.doesNotMatch(html, publicCopyIdentityPattern);
  assert.doesNotMatch(html, privatePathPattern);
  assert.doesNotMatch(html, /Starter Project|Building your site/);
});

test("keeps public source and built assets free of private runtime data", async () => {
  const [publicSource, builtClient, layout, packageJson] = await Promise.all([
    readPublicSource(),
    readBuiltClientText(),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  const shippedText = `${publicSource}\n${builtClient}`;

  assert.match(packageJson, /"name": "tsq-web"/);
  assert.match(layout, /TSQ — The Second Question/);
  assert.match(publicSource, /same Python engine and written to the same/);
  assert.match(publicSource, /Created on first session/);
  assert.match(publicSource, /No learner evidence yet/);
  assert.doesNotMatch(publicSource, /rel_[a-f0-9]{24}/i);
  assert.doesNotMatch(publicSource, /\b(?:recentSessions|objectiveEvidence|reviewQueue|authoringJobs|coverageGaps|taskCards)\b/);
  assert.doesNotMatch(publicSource, publicCopyIdentityPattern);
  assert.doesNotMatch(builtClient, identityPattern);
  assert.doesNotMatch(shippedText, privatePathPattern);
  assert.doesNotMatch(publicSource, rawBankShapePattern);
  assert.doesNotMatch(shippedText, /\b(?:s-104|job_048|claim_003)\b|Example learner/);
});

test("does not ship released question IDs, stems, or long answer options", async () => {
  const [publicSource, builtClient, releasedQuestions] = await Promise.all([
    readPublicSource(),
    readBuiltClientText(),
    readReleasedQuestions(),
  ]);
  const shippedText = `${publicSource}\n${builtClient}`;

  assert.ok(releasedQuestions.length > 0, "release corpus must contain questions");
  for (const question of releasedQuestions) {
    assert.equal(shippedText.includes(question.id), false, `released question ID leaked: ${question.id}`);
    assert.equal(shippedText.includes(question.stem), false, `released question stem leaked: ${question.id}`);
    for (const option of question.options ?? []) {
      if (option.text?.length >= 48) {
        assert.equal(
          shippedText.includes(option.text),
          false,
          `released answer option leaked: ${question.id}/${option.id}`,
        );
      }
    }
  }
});

test("keeps generated image assets free of vendor metadata", async () => {
  const imageUrls = [
    new URL("../public/icon.png", import.meta.url),
    new URL("../dist/client/icon.png", import.meta.url),
  ];

  for (const imageUrl of imageUrls) {
    const image = await readFile(imageUrl);
    assert.doesNotMatch(pngMetadataText(image), imageMetadataPattern, imageUrl.pathname);
  }
});

test("keeps mutation retries bound to their original commands", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(source, /answerCommands\.current\.get\(decisionId\)/);
  assert.match(source, /api\.answer\(decisionId, command\.input, command\.key\)/);
  assert.match(source, /disabled=\{submitting \|\| commandLocked\}/);
  assert.match(source, /Retry exact answer/);

  const startCall = source.indexOf("await api.startSession(");
  const nextCall = source.indexOf("await api.nextQuestion(", startCall);
  const startKeyRelease = source.indexOf("startKeys.current.delete(startScope)", startCall);
  assert.ok(startCall >= 0 && nextCall > startCall, "start flow must select a real question");
  assert.ok(
    startKeyRelease > nextCall,
    "the start idempotency key must survive until the initial question is selected",
  );

  assert.match(source, /reconciled = await api\.session\(session\.id\)/);
  assert.match(source, /setStudy\(priorStudy\)/);
  assert.match(source, /afterReceipt\(nextQuestion\)/);
  assert.match(source, /afterReceipt\(\(\) => finishSession\("completed"\)\)/);
});

test("keeps CLI-created sessions and optional telemetry interoperable", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(source, /api\.createLearner\(\{ learner_id: LEARNER_ID \}\)/);
  assert.doesNotMatch(source, /api\.createLearner\([^\n]*display_name/);
  assert.match(source, /summary\.topic_id \?\? summary\.root_concept_id/);
  assert.match(source, /name: summary\.target_name/);
  assert.match(source, /Prefer not to report/);
  assert.match(source, /confidence: optionId === null \? undefined : confidence \?\? undefined/);
  assert.match(source, /Question \{state\.session\.step\}/);

  const continuation = source.indexOf("const nextQuestion = useCallback");
  const nextMutation = source.indexOf("await api.nextQuestion(", continuation);
  const refreshedSession = source.indexOf("await api.session(session.id)", nextMutation);
  assert.ok(
    continuation >= 0 && nextMutation > continuation && refreshedSession > nextMutation,
    "the rendered session step must be fetched after the next-question mutation",
  );
});

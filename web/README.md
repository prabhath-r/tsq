# TSQ interface

The web workspace for **TSQ — The Second Question**.

It presents the exact active curriculum from the local TSQ engine and uses the
same adaptive learning operations and `tsq.db` as the command-line interface.
The browser talks to a loopback-only Python service; it never opens SQLite or
ships raw corpus shards.

## Development

```sh
npm install
npm run dev
```

That one command starts both the TSQ engine and the interface. The local
interface is available at `http://localhost:3000`, and changes made in either
the web interface or CLI are stored in the root `tsq.db`.

To run the two processes separately:

```sh
npm run dev:api
npm run dev:web
```

The browser uses same-origin `/api/v1` requests by default. A separately hosted
development API can be selected with `VITE_TSQ_API_URL` or
`NEXT_PUBLIC_TSQ_API_URL`; the local proxy target can be changed with
`TSQ_API_ORIGIN`. If the default ports are already occupied, choose exact local
ports without changing the data contract:

```sh
TSQ_API_PORT=8877 TSQ_WEB_PORT=3107 npm run dev
```

Production checks:

```sh
npm test
npm run lint
npm run typecheck
```

## Data boundary

- Curriculum and capacity surfaces come from the database's active immutable
  release rather than a browser snapshot.
- Learner, session, answer, progress, report, and trace operations go through
  the same engine used by the CLI.
- Raw question shards, answer-key inventories, database files, environment
  files, source-context echoes, and personal learner data are never bundled.

## Structure

- `app/page.tsx` — learner and operator workspaces
- `app/api.ts` — typed local TSQ API client
- `app/data.ts` — interface-only navigation types (no corpus snapshot)
- `app/globals.css` — responsive visual system
- `scripts/dev.mjs` — local API and interface supervisor
- `worker/index.ts` — deployment entry point
- `tests/rendered-html.test.mjs` — server-render and privacy checks

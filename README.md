# The Second Question

Adaptive learning engine with interchangeable CLI and local web interfaces.

Source checkout: `./start` begins a quiz; `./tsq --help` shows every command.

Browser interface (Python 3.12+ and Node 22+):

```sh
cd web
npm install   # first run only
npm run dev
```

Open `http://localhost:3000`. This one command starts the loopback-only Python
API and the web interface. Both the browser and CLI use the root `tsq.db`, the
same active corpus release, and the same adaptive engine. If the default ports
are occupied, use `TSQ_API_PORT=8877 TSQ_WEB_PORT=3107 npm run dev`.

`./serve` starts only the local JSON API when the frontend is run separately.

Installed package: `tsq start`; use `tsq --help` for every command.

Set `TSQ_DB=/path/to/tsq.db` to choose the database.

Container: `docker build -t tsq .`, then `docker run --rm -it --mount type=volume,src=tsq-data,dst=/data tsq start`.

Pushing a matching version tag such as `v0.2.0` publishes versioned Linux images to GitHub Container Registry after every check passes.

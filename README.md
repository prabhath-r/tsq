# The Second Question

Adaptive learning backend and CLI.

Source checkout: `./start` begins a quiz; `./tsq --help` shows every command.

Installed package: `tsq start`; use `tsq --help` for every command.

Set `TSQ_DB=/path/to/tsq.db` to choose the database.

Container: `docker build -t tsq .`, then `docker run --rm -it --mount type=volume,src=tsq-data,dst=/data tsq start`.

Pushing a matching version tag such as `v0.2.0` publishes versioned Linux images to GitHub Container Registry after every check passes.

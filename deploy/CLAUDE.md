# Deploy

LaunchAgent (macOS), systemd (Linux), and container manifests for serving.

## Layout

- `launchd/` — macOS user agents (`com.<org>.<service>.plist`)
- `systemd/` — Linux unit files (`<service>.service`, `<service>.socket`)
- Dockerfiles / compose: `../docker/`

## Conventions

- Plist `Label` mirrors the file name without `.plist`
- `WorkingDirectory` is absolute (no `~` expansion in launchd)
- Logs to `~/Library/Logs/<name>.{out,err}.log` (macOS), `/var/log/<name>.log` (Linux)
- All env vars in one block; document each in `../docs/`

## Installation

```bash
bash scripts/install-launchd.sh
```

Loads `launchd/com.electron.full-pipeline.plist.template` with
`${USER_HOME}` and `${REPO_DIR}` substituted for the current user.
The watchdog itself is documented in the F4L tree (cross-repo).

To uninstall:

```bash
bash scripts/install-launchd.sh --uninstall
```

## Anti-patterns

- Don't put secrets in plists/units — use a secrets manager or `.env` excluded from git
- Don't hardcode user paths — use template + install script
- Don't omit `KeepAlive` policy — choose `SuccessfulExit=false` for daemons
- Don't mix dev and prod configs in the same file — split by env

## Source of the launchd templates

The install script + plist template were instantiated from the
workspace-level templates at:
`~/Documents/Projets/_templates/launchd/` (promoted by N4 Task 8,
2026-05-10).

To bring future improvements from the workspace template back
into this project, diff against the source and merge selectively.

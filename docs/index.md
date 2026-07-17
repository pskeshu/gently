# Gently Documentation

Gently is a full-stack microscopy system: it joins sample preparation,
instrument control, perception, planning, storage, and operator workflows into
one inspectable loop.

This documentation is organized around that full stack rather than around a
single API layer.

## Start Here

- [Full Stack Microscopy](full-stack-microscopy.md): the system-integration map.
- [Try Without Hardware](guides/try-offline.md): run Gently offline.
- [Hardware Setup](guides/hardware-setup.md): connect a microscope/device layer.
- [Build a Plugin](guides/build-a-plugin.md): add organisms or hardware profiles.

## Architecture References

- [Sample and Hardware Domains](architecture/sample-hardware-domains.md)
- [Sample Tracking Metrics](architecture/sample-tracking-metrics.md)
- [Hardware Profile Template](architecture/hardware-profile-template.md)

## Build Locally

If MkDocs is installed, run:

```shell
mkdocs serve
```

The docs are plain Markdown, so they also remain readable directly in GitHub.

# Security policy

## Supported versions

SmartPick-VLA is pre-1.0 research software. Security fixes are applied to the
latest default branch and, after a tagged release exists, the most recent
minor-release line when maintainers can reproduce the issue. Older snapshots
and locally modified robot adapters are unsupported.

## Reporting a vulnerability

Use the repository's **Security** tab to open a private vulnerability report or
private security advisory. Do not open a public issue for a vulnerability that
could expose credentials, execute untrusted code, bypass action limits, or move
hardware unexpectedly.

Include:

- affected version or commit;
- operating system, Python and dependency versions;
- minimal reproduction and observed impact;
- whether ROS 2, an imported log, checkpoint, archive, or hardware adapter is
  involved;
- suggested mitigation, if known.

Do not include real credentials, proprietary logs, or personal images in the
report. Maintainers will acknowledge the report and coordinate disclosure when
the issue and fix have been assessed; no fixed response or remediation time is
guaranteed for this volunteer research project.

## Security-sensitive surfaces

### Checkpoints and serialized data

SmartPick-VLA v2 checkpoints store tensors and primitive metadata and are loaded
with PyTorch's restricted `weights_only=True` mode. Legacy or malformed files
that require unrestricted pickle loading are rejected. Still verify published
checksums and treat checkpoint, dataset, archive, and generated-media inputs as
untrusted; guard against path traversal and decompression bombs.

### Real logs

Logs can contain images, workstation paths, hostnames, IP addresses, ROS graph
details, serial numbers, and facility layout. Validate schema and size before
processing, resolve referenced files within the configured dataset root, and
redact sensitive content before publication.

### Configuration and command execution

YAML/JSON configuration is data, not code. Use safe parsers and do not evaluate
configuration strings or construct shell commands from untrusted values.
Generated artifact paths must remain within their declared output directory.

### Dependencies and media codecs

Image/video parsers and native simulation libraries handle complex untrusted
input. Keep dependencies updated, review automated dependency changes, and
avoid rendering unknown assets in a privileged process.

### ROS 2 and hardware

ROS 2 discovery and topics are not an authorization or security boundary. The
reference integration is dry-run only, with `hardware_enabled=false`. Do not
expose a ROS domain to an untrusted network, and do not rely on this software as
a safety PLC, emergency stop, collision monitor, or access-control system.

Any downstream hardware executor must authenticate/authorize commands where
appropriate, reject stale or malformed data, enforce independent limits, fail
closed on heartbeat loss, and integrate a physical emergency stop and guarded
enable procedure.

## Out of scope

- Safety certification or guarantees for physical robot operation.
- Vulnerabilities only present after intentionally removing documented safety
  checks.
- Secrets or proprietary data already committed by a third party with no link
  to this repository.
- Availability guarantees for long-running research jobs.

Physical safety concerns are still welcome as private reports when a software
defect could cause unexpected action output.

# Native tracking dependencies

The native runtime statically links ByteTrack-Eigen and Eigen behind
`cuajone::ByteTracker`. No upstream DLL or public upstream type crosses the project
adapter boundary.

## Provenance

| Dependency | Immutable source | Archive SHA-256 | License |
| --- | --- | --- | --- |
| ByteTrack-Eigen 2.1.0 | `cj-mills/byte-track-eigen` commit `a865158906f6138465668810a98ffd918d95f9a3` | `e5a075df5e8b4ed4bb7436ffe7fe0f4cee5c6a6663112d6a1c47a99ffb704d88` | MIT |
| Eigen 3.4.0 | `eigen/eigen` commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c` | `9eec4ec4e5e459b2f59dbbaa4280e1bb3ee61cccd8a7c0af0321d29d95fece9e` | Primarily MPL-2.0 |

Run `native/Provision-TrackingDependencies.ps1` to download, verify, and extract
both archives under the Git-ignored `.tools/native/dependencies` cache. CMake never
downloads dependencies and fails closed when a verified source receipt is absent.

## Project patch

`byte-track-eigen-cuajone.patch` is applied only after the ByteTrack archive hash
matches. It provides atomic process-wide IDs without constructor resets, bounded
retained tracks, explicit reset/count operations, immediate removed-track pruning,
a static-link export mode, and one-to-one deterministic rematching. The Cuajone
adapter adds input validation and detection-aligned IDs.

Eigen is compiled with `EIGEN_MPL2_ONLY`. Installer notices and the generated SPDX
SBOM record both static dependencies even though they add no runtime DLL.

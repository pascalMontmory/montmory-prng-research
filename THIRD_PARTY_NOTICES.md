# Rights and third-party notices

[Français](THIRD_PARTY_NOTICES.fr.md) · [English](THIRD_PARTY_NOTICES.md) · [Documentation FR/EN](docs/en/languages.md)

This repository preserves research materials attributed to Pascal Montmory and the third-party components identified below. It does **not** introduce an MIT, Apache or other general redistribution license for original Montmory code, data or documentation. Repository visibility and access are not a new license grant.

Frozen source bytes and third-party notices remain unchanged. Original contributions retain the rights reservation in [LICENSE](LICENSE).

Third-party materials retain their original terms, regardless of the status of the surrounding repository:

- **Random123**, copyright D. E. Shaw Research. The original notice appears in each vendored copy and in [third_party/Random123-LICENSE](third_party/Random123-LICENSE). The pinned upstream commit and file hashes are in [the provenance manifest](studies/audit-2026-10-21/evidence/random123-upstream-manifest.json).
- **TestU01 1.2.3**, copyright Pierre L'Ecuyer. Original source ZIPs included in the frozen testing kits retain their notices. The original [COPYING file](third_party/TestU01-COPYING) is also provided separately. Its terms distinguish private/research/academic/non-commercial use from commercial use; this repository does not replace those terms.
- **xoshiro256\*\***, David Blackman and Sebastiano Vigna. The source retains its public-domain dedication, permission notice and disclaimer in [xoshiro256starstar.c](studies/tacm-comparison-2026-10-21/deps/xoshiro256starstar.c).
- **JAX, NumPy, SciPy, pandas, matplotlib and openpyxl** are external dependencies, not bundled installations. Consult their upstream distributions for their respective license terms. JAX is an external framework used by the recorded TPU code, not a Montmory dependency ownership claim.

Historical filenames containing “Certified”, “official” or “DXSM” are retained for provenance. They do not establish external certification, official status of Montmory vectors, or identity with the standard PCG64DXSM algorithm. “Official vectors” in comparator reports refers to the cited upstream comparator vectors.

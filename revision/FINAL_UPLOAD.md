# TACL-11241 Official B-Decision Resubmission Pointer

**Canonical version: 2026-08-19.** The official TACL B-decision procedure requires
one anonymized PDF bundle. Do not choose files by searching for the newest-looking
filename, and do not upload the revision as a new submission in the TACL system.

## File to send

Send exactly this canonical bundle:

- `../geoprobe-release-artifacts/TACL-11241-major-revision-resubmission-bundle-20260819.pdf`
- SHA-256: `4e6b4f5137c07f25f0bda27c1e6859392a5fac0b9ef075b8c150fd6839f3dca1`

The bundle follows the official TACL resubmission skeleton and contains, in order:

1. the anonymized cover response and point-by-point reviewer letter;
2. the revised anonymized manuscript;
3. the anonymized original decision letter and reviews.

The editable in-repository source is `revision/RESUBMISSION_BUNDLE.pdf`.

## Official B-decision procedure

1. Email the bundle directly to `editors-in-chief@transacl.org`.
2. Use the subject: `reactivate TACL 11241`.
3. In the email body, provide:
   - the original Action Editor: Hai Zhao;
   - every author in publication order, last/family name first, with first name,
     middle name if applicable, email, and country of affiliation;
   - the same four-item bulleted change summary used in the cover response.
4. Attach the single resubmission-bundle PDF.
5. Do **not** upload the B-decision bundle onto the submission system unless the
   Editors-in-Chief explicitly instruct otherwise.
6. If no confirmation arrives within 5--7 days, inquire with the Editors-in-Chief.

## Supporting artifacts, not the official resubmission bundle

These files are retained under `internal-audit/` for audit/reproducibility but
are not substitutes for the single official PDF bundle:

- `internal-audit/TACL-11241-major-revision-main-20260819.pdf`
- `internal-audit/TACL-11241-major-revision-response-letter-20260819.pdf`
- `internal-audit/TACL-11241-major-revision-decision-letter-20260819.pdf`

The post-acceptance source/evidence candidate is

- `internal-audit/TACL-11241-major-revision-anonymous-final-20260819.tar.gz`

with a full preflight snapshot written beside it as
`internal-audit/ANONYMOUS_RELEASE_PREFLIGHT.json`.  The 20260816 tarball is a
historical baseline only.  Do not attach any tarball to the B-decision email
unless the Editors-in-Chief explicitly request it.

## Do not send

- Any `20260806`, `20260812`, `20260813`, `20260814`, `20260815`, or `20260816` artifact.
- Separate manuscript and response PDFs in place of the required bundle.
- The immutable June 1 submission PDF as the revised manuscript.
- Any internal planning, execution-log, server-audit, or identity-bearing file.
- Fig. 1 draw.io sources (`paper/figures/fig1_overview.drawio*`).

## Verification

The canonical bundle is A4, anonymous, and contains all three required parts.
The revised manuscript uses the official `tacl2021v1` style, contains no figure
or table on page 1, occupies PDF pages 1--12 before the references (References begin on PDF p.13; Limitations, Artifact Availability, and Ethics Statement finish on p.12), and
has its appendices after the references (PDF pp.15--17). See the SHA-256 file in the release
directory before sending.

# TACL 11241 final author checklist

**Snapshot:** 2026-08-19. Canonical deliverable: one official TACL B-decision
resubmission bundle.

## Official-format checks

- [x] Manuscript uses `\documentclass[11pt,a4paper]{article}` and the official
  `tacl2021v1` submission style.
- [x] The submission remains anonymous in the manuscript, response letter,
  decision/reviews copy, PDF metadata, and combined bundle.
- [x] The abstract begins in the first column on page 1; no figure or table appears
  on page 1.
- [x] The abstract is 250 words. The current TACL instructions impose placement
  and type-size requirements but do not state a separate abstract word maximum.
- [x] The revised manuscript body occupies PDF pages 1--12, with references beginning on PDF p.13, within the 10-page base limit plus the Action Editor's explicit
  one-to-two-page allowance. Limitations, Artifact Availability, and the Ethics
  Statement finish on p.12.
- [x] References are excluded from the content-page count.
- [x] Appendices follow the references, remain anonymous, and use the normal
  document font size rather than a global reduced font.
- [x] The appendix occupies PDF pages 15--17 (three pages total; Category 1 and Category 2 remain within their respective TACL page allowances), within the
  TACL appendix allowances.
- [x] The manuscript is A4, two-column, and includes the confidential header,
  line-number rulers, page numbers, and embedded fonts.

## Mandatory revision checks

- [x] Scale-dependent signature stability is audited and the universal
  fingerprint claim is withdrawn.
- [x] GeoVote is evaluated against same-pool majority, length, shortest-output,
  residualized, likelihood, and length-matched permutation controls.
- [x] CrossSteer is compared with conventional, sparse, sign-reversed,
  matched-random, and target-calibrated steering baselines under a locked protocol.
- [x] Paired confidence intervals, exact tests, and family-wise corrections are
  reported where applicable.
- [x] Complete frozen MATH-500 and SVAMP evaluations, long-context schedules,
  behavior telemetry, and the R1 readiness stop are reported.
- [x] Methods are consolidated in Section 3 with a notation table, intuition,
  and Algorithms 1--2.
- [x] The response letter contains a point-by-point reply to the Action Editor
  and Reviewers A--C, quotes each reviewer comment verbatim, and begins with a
  four-item bulleted change summary matching the decision letter.

## Final PDF checks

- [x] `paper/main.pdf`: revised anonymized manuscript (17 pages).
- [x] `revision/RESPONSE_LETTER.pdf`: anonymized eight-page point-by-point response.
- [x] `revision/ORIGINAL_DECISION_AND_REVIEWS_ANONYMIZED.pdf`: anonymized copy of
  the original decision and reviews.
- [x] `revision/RESUBMISSION_BUNDLE.pdf`: official single-PDF bundle containing
  the three required parts.
- [x] No Overfull boxes, unresolved references, placeholders, author identities,
  clipped text, or blank artifact pages were found in the canonical outputs.
- [x] Fig. 1 is the author-supplied generated schematic, print-prepared at 554 ppi
  in the PDF, with Ethics/Artifact retained on content p.12.
- [x] All pages of the manuscript, response, decision/reviews copy, and combined
  bundle were rebuilt together on 2026-08-19.
- [x] The anonymous-release preflight is ready (source tree, evidence, zero identity hits).
  The post-acceptance source/evidence candidate is
  `internal-audit/TACL-11241-major-revision-anonymous-final-20260819.tar.gz`.
  It is not part of the B-decision email.

## Author-only email actions

1. Attach only
   `TACL-11241-major-revision-resubmission-bundle-20260819.pdf` to the official
   reactivation email.
2. Send to `editors-in-chief@transacl.org` with subject
   `reactivate TACL 11241`.
3. Add the original Action Editor (Hai Zhao), the complete ordered author list,
   and for each author the first/middle/last name, email, and country of
   affiliation. These identity fields deliberately do not appear in the
   anonymous repository artifact.
4. Paste the four mandatory-change bullets from the response-letter cover into
   the email body.
5. Do not upload the B-decision bundle to the TACL system unless explicitly
   instructed by the Editors-in-Chief.
6. If no receipt confirmation arrives in 5--7 days, send a concise inquiry.
7. Confirm the two-month window from the dated decision email (the letter in
   the response notes a resend on 2026-08-12).
8. Do not launch additional outcome-seeking experiments before submission.  The two predeclared frozen audits (4,096-token MATH-500 budget audit; corrected R1 short-envelope study) have completed and are integrated; no further runs are planned.

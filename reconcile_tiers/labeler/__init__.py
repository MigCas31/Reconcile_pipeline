"""Generic human-in-the-loop labeler for binary/N-way geometry decisions.

A *run* groups cases sharing one decision_type (e.g. "roof-flat-vs-oblique").
Cases are generated offline and stored as JSONL; labels are appended
incrementally as humans adjudicate them in the viewer.
"""

from reconcile_tiers.labeler.schema import Case, CaseOption, Label, RunMeta

__all__ = ["Case", "CaseOption", "Label", "RunMeta"]

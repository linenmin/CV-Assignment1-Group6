# bad image probe

- source: exp_088 crop pipeline selection_audit.csv
- scope: test only
- candidates: selection_source == crop_fallback
- no hash / dhash anchor heuristics
- `priority_candidates.csv`: fallback candidate list
- `all_test_audit.csv`: full test audit rows from exp_088 preprocessing
- `images/`: rendered raw + selected crop panels

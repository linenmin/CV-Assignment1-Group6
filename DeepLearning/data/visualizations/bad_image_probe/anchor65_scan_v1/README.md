# bad image probe

- anchor strategy: known bad train ids
- score = exact hash + dhash similarity + low gradient + no-face penalty
- `priority_candidates.csv`: ranked inspection list
- `all_stats.csv`: full split statistics
- `images/`: rendered audit panels
- anchors=[65]
- grad_p10=3.308820
- grad_p30=4.579502

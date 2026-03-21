import unittest

import pandas as pd


from dl_pipeline.data.split import make_stratified_holdout_split


class SplitTests(unittest.TestCase):
    def test_stratified_holdout_keeps_all_rows_and_has_no_overlap(self):
        rows = []
        for class_id in [0, 1, 2]:
            for i in range(10):
                rows.append({"id": f"{class_id}_{i}", "class": class_id, "path": f"img_{class_id}_{i}.png"})
        frame = pd.DataFrame(rows)

        train_df, val_df = make_stratified_holdout_split(frame, val_ratio=0.2, random_state=42)

        self.assertEqual(len(train_df) + len(val_df), len(frame))
        self.assertEqual(set(train_df["id"]).intersection(set(val_df["id"])), set())
        self.assertEqual(sorted(frame["id"].tolist()), sorted(pd.concat([train_df, val_df])["id"].tolist()))
        self.assertEqual(val_df["class"].value_counts().to_dict(), {0: 2, 1: 2, 2: 2})


if __name__ == "__main__":
    unittest.main()

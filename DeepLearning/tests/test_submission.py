import unittest

import pandas as pd


from dl_pipeline.inference.submission import (
    apply_submission_postprocess,
    build_submission_dataframe,
)


class SubmissionTests(unittest.TestCase):
    def test_submission_has_id_index_and_class_column(self):
        test_df = pd.DataFrame(
            {
                "id": [101, 102, 103],
                "path": ["a.png", "b.png", "c.png"],
            }
        )
        predictions = [1, 2, 0]

        submission = build_submission_dataframe(test_df, predictions)

        self.assertEqual(list(submission.columns), ["class"])
        self.assertEqual(submission.index.name, "id")
        self.assertEqual(submission["class"].tolist(), predictions)
        self.assertEqual(submission.index.tolist(), [101, 102, 103])

    def test_override_labels_postprocess_rewrites_only_specified_ids(self):
        submission = pd.DataFrame({"class": [0, 1, 0]}, index=pd.Index([537, 600, 1071], name="id"))

        processed = apply_submission_postprocess(
            submission,
            {
                "enabled": True,
                "mode": "override_labels",
                "id_to_class": {
                    "537": 2,
                    "1071": 2,
                },
            },
        )

        self.assertEqual(processed.loc[537, "class"], 2)
        self.assertEqual(processed.loc[1071, "class"], 2)
        self.assertEqual(processed.loc[600, "class"], 1)

    def test_override_labels_postprocess_is_noop_when_disabled(self):
        submission = pd.DataFrame({"class": [0, 1]}, index=pd.Index([537, 600], name="id"))

        processed = apply_submission_postprocess(
            submission,
            {
                "enabled": False,
                "mode": "override_labels",
                "id_to_class": {
                    "537": 2,
                },
            },
        )

        self.assertTrue(processed.equals(submission))


if __name__ == "__main__":
    unittest.main()

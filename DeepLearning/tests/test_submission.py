import unittest

import pandas as pd


from dl_pipeline.inference.submission import build_submission_dataframe


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


if __name__ == "__main__":
    unittest.main()

import unittest


from dl_pipeline.preprocess.build_processed_dataset import build_processed_record


class PreprocessRecordTests(unittest.TestCase):
    def test_build_processed_record_keeps_class_label_when_present(self):
        row = {
            "id": 7,
            "source_path": "data/raw/train/train_7.npy",
            "class": 2,
        }

        record = build_processed_record(
            row=row,
            output_path="data/processed/train/7.png",
        )

        self.assertEqual(record["id"], 7)
        self.assertEqual(record["source_path"], "data/raw/train/train_7.npy")
        self.assertEqual(record["image_path"], "data/processed/train/7.png")
        self.assertEqual(record["class"], 2)


if __name__ == "__main__":
    unittest.main()

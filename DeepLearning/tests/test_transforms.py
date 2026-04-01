import unittest

from dl_pipeline.data.transforms import build_eval_transform, build_train_transform


class TransformTests(unittest.TestCase):
    def test_train_transform_can_disable_horizontal_flip(self):
        transform = build_train_transform(image_size=224, use_horizontal_flip=False)
        transform_names = [item.__class__.__name__ for item in transform.transforms]

        self.assertNotIn("HorizontalFlip", transform_names)
        self.assertIn("ColorJitter", transform_names)

    def test_train_transform_can_use_affine_instead_of_shift_scale_rotate(self):
        transform = build_train_transform(image_size=224, use_horizontal_flip=False, use_affine=True)
        transform_names = [item.__class__.__name__ for item in transform.transforms]

        self.assertIn("Affine", transform_names)
        self.assertNotIn("ShiftScaleRotate", transform_names)

    def test_eval_transform_can_use_face_recognition_normalization(self):
        transform = build_eval_transform(image_size=112, normalization="face")
        normalize = next(item for item in transform.transforms if item.__class__.__name__ == "Normalize")

        self.assertEqual(tuple(normalize.mean), (0.5, 0.5, 0.5))
        self.assertEqual(tuple(normalize.std), (0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()

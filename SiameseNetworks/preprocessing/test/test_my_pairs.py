import pytest
import numpy as np
import Unsupervised_line_segmentation_siamese_dnn.my_way.preprocessing.my_pairs as pairs


@pytest.mark.parametrize("img",
                         [np.zeros((500, 400)), np.zeros((800, 900)), np.zeros((1354, 2769))])
@pytest.mark.parametrize("patch_size", [120, 130, 150, 200])
class TestGetPosition:

    def test_get_position_too_big_Y_axis(self, img, patch_size):
        p1_pos, p2_pos = pairs.get_position(img, patch_size)
        assert p2_pos[0] + patch_size < img.shape[0]

    def test_get_position_too_big_X_axis(self, img, patch_size):
        p1_pos, p2_pos = pairs.get_position(img, patch_size)
        assert p2_pos[1] + patch_size < img.shape[1]

    def test_get_position_too_small_X_axis(self, img, patch_size):
        p1_pos, p2_pos = pairs.get_position(img, patch_size)
        assert p1_pos[0] > 0

    def test_get_position_too_small_Y_axis(self, img, patch_size):
        p1_pos, p2_pos = pairs.get_position(img, patch_size)
        assert p1_pos[1] > 0


@pytest.mark.parametrize("small_img_on_Y",
                         [np.zeros((200, 400)), np.zeros((120, 900)), np.zeros((240, 2769))])
@pytest.mark.parametrize("patch_size", [120, 130, 150, 200])
class TestGetPosition_on_Y:

    def test_too_small_img(self, small_img_on_Y, patch_size):
        with pytest.raises(AssertionError):
            pairs.get_position(small_img_on_Y, patch_size)


@pytest.mark.parametrize("small_img_on_X",
                         [np.zeros((460, 80)), np.zeros((420, 100)), np.zeros((940, 120))])
@pytest.mark.parametrize("patch_size", [120, 130, 150, 200])
class TestGetPosition_on_X:

    def test_too_small_img(self, small_img_on_X, patch_size):
        with pytest.raises(AssertionError):
            pairs.get_position(small_img_on_X, patch_size)

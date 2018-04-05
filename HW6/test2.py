from HW6.question2 import *


def test_wrap():
    mat = np.arange(0, 9).reshape((3, 3)).astype(np.uint8)
    im = wrap(mat, np.zeros(mat.shape), np.zeros(mat.shape))
    assert np.sum(im != mat) == 0

    im = wrap(mat, np.zeros(mat.shape), np.ones(mat.shape))
    assert np.sum(im[:, 0:-1] != mat[:, 1:]) == 0

    im = wrap(mat, np.ones(mat.shape), np.zeros(mat.shape))
    assert np.sum(im[0:-1, :] != mat[1:, :]) == 0


if __name__ == '__main__':
    test_wrap()

import unittest
import tracker

class MyTestCase(unittest.TestCase):

    def test_merge_point(self):
        self.assertEqual((2,3), tracker.merge_point([(1,2), (3, 4)]))

    def test_dist(self):
        self.assertEqual(5, tracker.dist((1, 1), (4, 5)))

    def test_argsort(self):
        self.assertEqual([1, 0], tracker.argsort((8, 4)))


if __name__ == '__main__':
    unittest.main()

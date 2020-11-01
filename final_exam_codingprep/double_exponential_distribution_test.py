import unittest

from double_exponential_distribution import Solution


class TestSolution(unittest.TestCase):

    def test_xis0(self):
        sol = Solution.solution(0.)
        self.assertAlmostEqual(sol, 0.8996324353165482, places=2)

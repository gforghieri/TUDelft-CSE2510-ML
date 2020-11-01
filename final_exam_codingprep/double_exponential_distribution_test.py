import unittest
from solution import Solution
import weblabTestRunner


class TestSolution(unittest.TestCase):

  def test_xis0(self):
    sol = Solution.solution(0.)
    self.assertAlmostEqual(sol, 0.8996324353165482, places=2)

if __name__ == "__main__":
  unittest.main(testRunner=weblabTestRunner.TestRunner)

import unittest
class TestSimple(unittest.TestCase):
    def setUp(self):
        self.a = 1
        self.b = 2
       
    def test_addition(self):
        """Test basic addition"""
        result = self.a + self.b
        self.assertEqual(result, 3)
    def test_subtraction(self):
        """Test basic subtraction"""
        result = self.b - self.a
        self.assertEqual(result, 1)
       
if __name__ == '__main__':
   unittest.main()
import unittest


class Testing(unittest.TestCase):

    def test_string(self): ## your code goes from here
        a = 'some'
        b = 'some'
        self.assertEqual(a, b) ## use the appropriate assert depending on your test

    def test_boolean(self):
        a = True
        b = True
        self.assertEqual(a, b)


if __name__ == '__main__':  #this line means that the test will run if you run the code from this file
    unittest.main()

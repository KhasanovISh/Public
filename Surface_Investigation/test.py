import os
import shutil
import tempfile
import unittest
import zipfile


from helper import unzip_file


class TestUnzipFile(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to extract the zip file to
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Delete the temporary directory
        shutil.rmtree(self.test_dir)

    def test_unzip_file(self):
	"""
	In this example, the zipfile module creates a test zip file with three text files inside it. then the zip file extracted to the temporary directory using unzip_file() and checks if all three files were extracted correctly using os.path.exists() and self.assertTrue().
	"""
        # Generate test zip file
        zip_path = os.path.join(self.test_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr('file1.txt', 'This is file 1.')
            zip_file.writestr('file2.txt', 'This is file 2.')
            zip_file.writestr('file3.txt', 'This is file 3.')

        # Extract zip file to test directory
        extract_path = self.test_dir
        unzip_file(zip_path, extract_path)

        # Check if all files were extracted correctly
        expected_files = ['file1.txt', 'file2.txt', 'file3.txt']
        for file in expected_files:
            file_path = os.path.join(extract_path, file)
            self.assertTrue(os.path.exists(file_path), f"File {file} was not extracted correctly.")

if __name__ == '__main__':
    unittest.main()

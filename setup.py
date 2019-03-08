import sys
from cx_Freeze import setup, Executable
import os
import os.path
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR,'tcl','tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

files = {"include_files": ["checkersAnalyzer.py"],"packages":["numpy"]}


setup(
	name="Any Name",
	version="3.1",
	description = "Warcaby",
    options = {"build_exe": files},
	executables = [Executable("main.py",base="Win32GUI")])
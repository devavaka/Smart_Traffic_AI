import os
import sys
import subprocess

if __name__ == '__main__':
    # Get the path to the inner directory where the files actually live
    inner_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic project')
    
    # Change the current working directory so the script can find its assets
    os.chdir(inner_dir)
    
    # Run the real main.py as a separate clean process
    sys.exit(subprocess.call([sys.executable, 'main.py'] + sys.argv[1:]))

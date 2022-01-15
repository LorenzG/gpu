import numpy as np
from mock import patch
import test
from contextlib import ExitStack

# import inspect
# aa = inspect.getmodule(test.abs).__name__
# aaa.__module__

with patch('test.np', test):
    print('a')
    print(test.aaa())

#     print(aaa())
    
# with ExitStack() as stack:
#     for mgr in [patch('test2.np', test), patch('test.numpy')]:
#         stack.enter_context(mgr)
#     print(aaa())

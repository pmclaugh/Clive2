import pytest
from primitives import BoxStack
from constants import ONES, ZEROS


# @pytest.mark.unittest
# def test_box_stack_basics(unit_box):
#     bs = BoxStack()
#
#     assert bs.size == 0
#     assert bs.head is None
#
#     bs.push(unit_box)
#
#     assert bs.size == 1
#     assert bs.head is not None
#
#     b = bs.pop()
#     assert bs.size == 0
#     assert bs.head is None
#     # todo would be nice to have equality for Boxes
#     assert (b.min == ZEROS).all()
#     assert (b.max == ONES).all()
#
#     b = bs.pop()
#     assert b is None
#     assert bs.size == 0
#     assert bs.head is None

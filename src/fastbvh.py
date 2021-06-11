import numba
from utils import timed
from constants import *
from primitives import TreeBox, FastBox, unit, point


@numba.njit
def divide_box(box: TreeBox, axis, fraction: float):
    delta = (box.max - box.min) * axis
    return TreeBox(box.min, box.max - delta * fraction), TreeBox(box.min + delta * (1 - fraction), box.max)


@numba.njit
def surface_area_heuristic(l: TreeBox, r: TreeBox):
    return TRAVERSAL_COST + sah_component(l) + sah_component(r)


@numba.njit
def sah_component(box: TreeBox):
    return len(box.triangles) * INTERSECT_COST * box.surface_area()


@numba.njit
def surface_area(mins, maxes):
    span = maxes - mins
    return 2 * (span[0] * span[1] + span[1] * span[2] + span[2] * span[0])


@numba.njit
def volume(b: TreeBox):
    dims = b.max - b.min
    return dims[0] * dims[1] * dims[2]


@timed
# @numba.njit
def fastBVH(triangles):
    root = construct_fastBVH(triangles)
    return flatten_fastBVH(root)


@numba.njit
def flatten_fastBVH(root: TreeBox):
    flat_boxes = numba.typed.List()
    flat_triangles = numba.typed.List()
    emitters = numba.typed.List()
    box_queue = [root]
    while box_queue:
        box = box_queue[0]
        fast_box = FastBox(box.min, box.max)
        if box.right is not None and box.left is not None:
            fast_box.left = len(flat_boxes) + len(box_queue)
            box_queue.append(box.left)
            box_queue.append(box.right)
            # right will always be left + 1, this is how we signal inner node in fast traverse
            fast_box.right = 0
        else:
            fast_box.left = len(flat_triangles)
            for triangle in box.triangles:
                if triangle.emitter:
                    emitters.append(triangle)
                flat_triangles.append(triangle)
            fast_box.right = len(flat_triangles)
            # so now triangles[left:right] is the triangles in this box
        flat_boxes.append(fast_box)
        box_queue = box_queue[1:]
    return flat_boxes, flat_triangles, emitters


# @numba.njit
@timed
def construct_fastBVH(triangles):
    start_box = bound_triangles(triangles)

    stack = [start_box]
    while len(stack) > 0:
        box = stack.pop()
        if (len(box.triangles) <= MAX_MEMBERS) or len(stack) > MAX_DEPTH:
            continue

        sah_object, l_object, r_object = object_split(box)
        # todo - alpha test. compare (surface area of intersection of l_object + r_object) / surface area of root
        #  to some small alpha. if under, don't bother doing (costly) spatial split
        # sah_spatial, l_spatial, r_spatial = spatial_split(box)

        # if sah_spatial < sah_object:
        #     l, r = l_spatial, r_spatial
        # else:
        #     l, r = l_object, r_object

        l, r = l_object, r_object

        if r is not None:
            box.right = r
            r.parent = box
            stack.append(r)
        if l is not None:
            box.left = l
            l.parent = box
            stack.append(l)

    return start_box


@timed
def object_split(box: TreeBox):
    best_sah = np.inf
    best_split = None
    best_axis = None

    triangle_mins = np.array([t.mins for t in box.triangles])
    triangle_maxes = np.array([t.maxes for t in box.triangles])
    triangle_centers = (triangle_mins + triangle_maxes) / 2

    for axis in [UNIT_X, UNIT_Y, UNIT_Z]:
        axis_only = np.dot(triangle_centers, axis)
        axis_sorted = np.argsort(axis_only)

        ltr_maxes = np.maximum.accumulate(triangle_maxes[axis_sorted])
        ltr_mins = np.minimum.accumulate(triangle_mins[axis_sorted])
        rtl_maxes = np.maximum.accumulate(triangle_maxes[axis_sorted[::-1]])[::-1]
        rtl_mins = np.minimum.accumulate(triangle_mins[axis_sorted[::-1]])[::-1]

        for i, left_max in enumerate(ltr_maxes[:-1]):
            left_min = ltr_mins[i]
            right_max = rtl_maxes[i + 1]
            right_min = rtl_mins[i + 1]
            sah = surface_area(left_min, left_max) * i + surface_area(right_min, right_max) * (len(ltr_maxes) - i)
            if sah < best_sah:
                best_sah = sah
                best_split = (i, left_min, left_max, right_min, right_max)
                best_axis = axis

    i, left_min, left_max, right_min, right_max = best_split
    left_triangles = numba.typed.List()
    right_triangles = numba.typed.List()
    sort_keys = np.argsort(np.dot(triangle_centers, best_axis))
    for j, key in enumerate(sort_keys):
        if j <= i:
            left_triangles.append(box.triangles[sort_keys[j]])
        else:
            right_triangles.append(box.triangles[sort_keys[j]])

    left_box = TreeBox(left_min, left_max)
    left_box.triangles = left_triangles
    right_box = TreeBox(right_min, right_max)
    right_box.triangles = right_triangles

    print(best_axis, 'produced best split:', best_split)
    print('sah', best_sah)
    print('left vol:', volume(left_box) / volume(box), 'right vol:', volume(right_box) / volume(box))
    print('left count:', len(left_box.triangles), 'right count:', len(right_box.triangles))

    # todo make this a separate test
    test_left = bound_triangles(left_box.triangles)
    test_right = bound_triangles(right_box.triangles)
    assert np.equal(test_left.min, left_box.min).all()
    assert np.equal(test_left.max, left_box.max).all()
    assert np.equal(test_right.min, right_box.min).all()
    assert np.equal(test_right.max, right_box.max).all()

    return best_sah, left_box, right_box


@numba.njit
def bound_triangles(triangles):
    box = TreeBox(INF, NEG_INF)
    for triangle in triangles:
        box.extend(triangle)
    box.triangles = triangles
    return box


@numba.njit
def partition_box(box: TreeBox, axis, num_partitions):
    delta = (box.max - box.min) * axis
    partitions = []
    delta_step = delta / num_partitions
    part_min = box.min
    part_max = box.max * (1 - axis) + box.min * axis + delta_step
    for _ in range(num_partitions):
        partitions.append(TreeBox(part_min, part_max))
        part_min = part_min + delta_step
        part_max = part_max + delta_step
    return partitions


# @numba.njit
@timed
def spatial_split(parent_box: TreeBox):
    best_sah = np.inf
    best_split = None
    best_axis = None

    for axis_index, axis in enumerate([UNIT_X, UNIT_Y, UNIT_Z]):

        boxes = partition_box(parent_box, axis, SPATIAL_SPLITS)
        bags = [TreeBox(INF, NEG_INF) for _ in range(len(boxes))]

        ins = np.zeros(len(boxes), dtype=np.int64)
        outs = np.zeros_like(ins)

        # todo: this is really memory inefficient
        in_lists = [[] for _ in range(len(boxes))]
        out_lists = [[] for _ in range(len(boxes))]

        left_bound = parent_box.min[axis_index]
        right_bound = parent_box.max[axis_index]
        span = right_bound - left_bound

        for triangle in parent_box.triangles:
            # unit vectors and lengths for each edge (todo: cache these)
            v0 = triangle.v0
            v1 = triangle.v1
            v2 = triangle.v2
            u0 = unit(v1 - v0)
            t0 = np.linalg.norm(v1 - v0)
            u1 = unit(v2 - v1)
            t1 = np.linalg.norm(v2 - v1)
            u2 = unit(v0 - v2)
            t2 = np.linalg.norm(v0 - v2)

            v = [v0[axis_index], v1[axis_index], v2[axis_index]]
            least = max(0, min(int(SPATIAL_SPLITS * (min(v) - left_bound) / span), len(ins) - 1))
            most = max(0, min(int(SPATIAL_SPLITS * (max(v) - left_bound) / span), len(outs) - 1))
            ins[least] += 1
            in_lists[least].append(triangle)
            outs[most] += 1
            out_lists[most].append(triangle)
            for box, bag in zip(boxes[least:most + 1], bags[least:most + 1]):
                for origin, direction, t_max in zip([v0, v1, v2], [u0, u1, u2], [t0, t1, t2]):
                    if box.contains_point(origin):
                        bag.extend_point(origin)
                    else:
                        # solve origin + direction * t = bound
                        # for left and right bounds
                        o = origin[axis_index]
                        d = direction[axis_index]
                        if d == 0:
                            continue
                        for bound in [box.min, box.max]:
                            b = bound[axis_index]
                            t = (b - o) / d
                            if 0 <= t <= t_max:
                                clip_point = origin + direction * t
                                # clamp to this box
                                clip_point = np.minimum(np.maximum(box.min, clip_point), box.max)
                                bag.extend_point(clip_point)


        in_sums = np.add.accumulate(ins)
        out_sums = np.add.accumulate(outs[::-1])[::-1]
        mins = np.array([b.min for b in bags])
        maxes = np.array([b.max for b in bags])
        maxes_ltr = np.maximum.accumulate(maxes)
        mins_ltr = np.minimum.accumulate(mins)
        maxes_rtl = np.maximum.accumulate(maxes[::-1])[::-1]
        mins_rtl = np.minimum.accumulate(mins[::-1])[::-1]
        for i in range(SPATIAL_SPLITS - 1):
            # i is the index of the rightmost left bag, i + 1 is the leftmost right bag
            sah = surface_area(mins_ltr[i], maxes_ltr[i]) * in_sums[i] \
                  + surface_area(mins_rtl[i + 1], maxes_rtl[i + 1]) * out_sums[i + 1]
            if sah < best_sah:
                best_sah = sah
                best_l, best_r = TreeBox(mins_ltr[i], maxes_ltr[i]), TreeBox(mins_rtl[i + 1], maxes_rtl[i + 1])
                left_triangles = numba.typed.List()
                right_triangles = numba.typed.List()
                for in_list in in_lists[:i + 1]:
                    for t in in_list:
                        left_triangles.append(t)
                for out_list in out_lists[i + 1:]:
                    for t in out_list:
                        right_triangles.append(t)
                best_l.triangles = left_triangles
                best_r.triangles = right_triangles
                best_split = best_l, best_r
                best_axis = axis

    left, right = best_split

    print(best_axis, 'produced best split:', left.min, left.max, right.min, right.max)
    print('sah', best_sah)
    print('left vol:', volume(left) / volume(parent_box), 'right vol:', volume(right) / volume(parent_box))
    print('left count:', len(left.triangles), 'right count:', len(right.triangles))

    return best_sah, left, right


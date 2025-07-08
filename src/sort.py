import numpy as np
import metalcompute as mc
from random import shuffle

dev = mc.Device()
input_data = [12, 3, 7, 1, 9, 4, 11, 2]
shuffle(input_data)
data = np.array(input_data, dtype=np.int32)
buf = dev.buffer(data)  # copy data to GPU memory

source = """
#include <metal_stdlib>
using namespace metal;

kernel void bitonic_sort(device int       *data        [[buffer(0)]],
                         constant uint    &stage       [[buffer(1)]],
                         constant uint    &passOfStage [[buffer(2)]],
                         uint              id          [[thread_position_in_grid]])
{
    /* distance between the two elements compared by this thread */
    uint pairDistance = 1u << (passOfStage - 1);      // 2^(p‑1)

    /* size of the whole bitonic block being merged in this stage (2^stage) */
    uint blockWidth   = 1u <<  stage;                 // 2^stage

    /* indices of the two elements that this thread will compare */
    uint leftId  = (id / pairDistance) * pairDistance * 2u + (id % pairDistance);
    uint rightId = leftId + pairDistance;

    /* ascending for the first half of each block, descending for the second */
    bool ascending = ((id & (blockWidth >> 1)) == 0u);   // test bit (stage‑1)

    /* compare–swap */
    int a = data[leftId];
    int b = data[rightId];
    if ((ascending && a > b) || (!ascending && a < b)) {
        data[leftId]  = b;
        data[rightId] = a;
    }
}

"""

sort_fn = dev.kernel(source).function("bitonic_sort")

n = data.size
log_n = int(np.log2(n))
for stage in range(1, log_n + 1):
    for passOfStage in range(stage, 0, -1):
        sort_fn(n // 2, buf, np.uint32(stage), np.uint32(passOfStage))

sorted_data = np.frombuffer(buf, dtype=np.int32)
print(sorted_data.tolist())  # → [1, 2, 3, 4, 7, 9, 11, 12]

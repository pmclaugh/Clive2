# Clive2
Bidirectional Path Tracing implemented in Python, relying heavily on Numba for performance. This is a work in progress and may be unstable, incorrect, etc. especailly as I'm just getting started with it. Performance is, well, poor. But for single-threaded, cpu-only, it's actually pretty decent. Eventually, I hope to use this as a reference renderer when implementing a GPU and/or cloud renderer.

Working:
- loading obj files
- ray casting
- collision
- collision acceleration with simple spatial-split BVH
- path generation
- importance sampling
- multiple importance sampling (Balance)
- combining bidirectional paths (except for t==0 and t==1 which are planned but NYI)
- diffuse surfaces
- specular surfaces

Known bugs:
- Sample 0 does not display correctly
- Bidirectional cannot currently handle specular surfaces

Planned features:
- textures
- bump mapping
- normal smoothing
- transmissive materials
- glossy materials

Optimistic possible features:
- Metropolis light transport
- GPU suport
- Cloud rendering

There were some sample images here but they were terrible. Better renders coming soon.

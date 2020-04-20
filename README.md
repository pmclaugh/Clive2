# Clive2
Bidirectional Path Tracing implemented in Python, relying heavily on Numba for performance. This is a work in progress and may be unstable, incorrect, etc. especailly as I'm just getting started with it. In the future I will try to keep uncertain code out of master, but currently there is only master.

Working:
- loading obj files
- ray casting
- collision
- path generation
- importance sampling
- combining bidirectional paths

Possibly not working:
- multiple importance sampling
- proper handling of the geometry term

Known bugs:
- Sample 0 does not display correctly
- Some light paths end up with negative values in ray.color, bug is presumably in the BRDF routines
- Bidirectional images appear dark and kind of washed-out compared to unidirectional (possibly related to above)
- Bidirectional cannot currently handle specular surfaces

Planned features:
- textures
- bump mapping
- normal smoothing
- transmissive materials
- glossy materials
- Metropolis light transport

Camera-only tracing, ~200 samples per pixel:


![unidirectional](https://github.com/pmclaugh/Clive2/blob/master/resources/unidirectional_example.jpg)


Bidirectional, 5 samples per pixel:


![bidirectional_low_sample](https://github.com/pmclaugh/Clive2/blob/master/resources/bidirectional_low_sample.png)

# Clive2
Bidirectional Path Tracing orchestrated with Python and run with Metal. 
This is a work in progress and may be unstable, incorrect, etc. 
Performance is... fine? Other than a simple BVH implementation, optimization has not been a priority yet. 

Working:
- Bidirectional Path Tracing in Metal
- Importance sampling, balance heuristic
- Loading OBJ + PLY files
- BVH acceleration structure
- Diffuse and specular materials, including transmission
- GGX BRDF

Note: there's a lot of junk left over from an earlier Python-only implementation. I will clean this up eventually, but for now you really just want to look at `trace.metal`.

![2024-09-28_09-37-36](https://github.com/user-attachments/assets/b3a1b770-27ec-4a46-a173-de14612720cd)

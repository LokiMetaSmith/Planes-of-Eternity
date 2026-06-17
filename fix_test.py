import sys
import re

filepath = "reality-engine/tests/host_test.rs"
with open(filepath, 'r') as f:
    content = f.read()

# We changed a uniform size/type or the structure of `RealityUniform`. We must fix it.

content = content.replace(
    """        reality_uniform.proj_params[0] = [0.0, 1.0, 0.0, 0.0];""",
    """        reality_uniform.proj_params[0] = [0.0, 1.0, 0.0, 0.0];
        reality_uniform.proj_pos_fid[0][3] = 1.0;"""
)

with open(filepath, 'w') as f:
    f.write(content)

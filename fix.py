with open('reality-engine/src/shader_dit_compute.wgsl', 'r') as f:
    text = f.read()
text = "enable subgroups;\n" + text
with open('reality-engine/src/shader_dit_compute.wgsl', 'w') as f:
    f.write(text)

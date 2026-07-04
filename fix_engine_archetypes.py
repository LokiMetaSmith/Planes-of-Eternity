import sys

def main():
    engine_path = "reality-engine/src/engine.rs"
    with open(engine_path, "r") as f:
        content = f.read()

    # Specifically we are trying to add Fractal to the second match
    target = """                                    crate::reality_types::RealityArchetype::WildWest => {
                                        [0.8, 0.5, 0.2, 1.0]
                                    }
                                };"""

    replacement = """                                    crate::reality_types::RealityArchetype::WildWest => {
                                        [0.8, 0.5, 0.2, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Fractal => {
                                        [1.0, 0.5, 0.0, 1.0]
                                    }
                                };"""

    if target in content:
        content = content.replace(target, replacement)
        with open(engine_path, "w") as f:
            f.write(content)

if __name__ == "__main__":
    main()

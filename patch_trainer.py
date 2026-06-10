"""Minimal patch: add model_converters support to upstream torchtitan trainer.py."""

trainer_path = "/home/amd/shekhar/torchtitan_upstream/torchtitan/trainer.py"

with open(trainer_path) as f:
    content = f.read()

# 1. Add import for ModelConvertersContainer
if "ModelConvertersContainer" not in content:
    content = content.replace(
        "from torchtitan.components.quantization.utils import has_quantization",
        "from torchtitan.components.quantization.utils import has_quantization\nfrom torchtitan.protocols.model_converter import ModelConvertersContainer"
    )

# 2. Add model_converters field to Trainer.Config
if "model_converters" not in content:
    content = content.replace(
        'profiling: ProfilingConfig = field(default_factory=ProfilingConfig)',
        'profiling: ProfilingConfig = field(default_factory=ProfilingConfig)\n\n'
        '        model_converters: ModelConvertersContainer.Config = field(\n'
        '            default_factory=ModelConvertersContainer.Config\n'
        '        )'
    )

# 3. Add converter application after model is built and parallelized
# Find: "model = model_config.build()" and add converters after
# But more carefully: we need to find where model is built then add converter block
old_block = """        # Build the model
        with TimerLogger(logger.info, "Building model"), self._init_ctx():
            model = model_config.build()"""

new_block = """        # Apply model converters to config BEFORE building
        model_converters = config.model_converters.build(
            parallel_dims=parallel_dims,
            model_compile_enabled=model_spec.model.compile.enable
            if hasattr(model_spec.model, "compile") else False,
        )
        from torchtitan.protocols.model_converter import ModelConverter
        for mc in getattr(model_converters, "converters", []):
            if hasattr(mc, "convert") and callable(mc.convert):
                mc.convert(model_config)

        # Build the model
        with TimerLogger(logger.info, "Building model"), self._init_ctx():
            model = model_config.build()

        # Apply model-level converters (post-build)
        model_converters.convert(model)"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        print("✓ Added model_converters application")
    else:
        print("✗ Could not find model build block")
        # Try finding just model_config.build()
        import re
        for m in re.finditer(r"model_config\.build\(\)", content):
            ctx = content[m.start()-200:m.start()+50]
            print(f"Found at {m.start()}: ...{ctx[-80:]}...")
        
# 4. Add post_optimizer hook
if "post_optimizer_hook" not in content:
    # Find optimizers setup and add hook
    content = content.replace(
        "self.optimizers = config.optimizer.build(",
        "self.model_converters = config.model_converters.build(\n"
        "            parallel_dims=parallel_dims,\n"
        "            model_compile_enabled=config.compile.enable,\n"
        "        )\n"
        "        self.optimizers = config.optimizer.build(",
        1  # only first occurrence (the real one, not the string in imports)
    )

with open(trainer_path, "w") as f:
    f.write(content)
print("Patched trainer.py")

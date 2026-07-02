from filegraph_agents import FGAConfig, FGARuntime, LiteLLMModel

config = FGAConfig.from_env()
runtime = FGARuntime(
    root=".",
    config=config,
    model=LiteLLMModel(config),
)

print(runtime.run("Inspect the project and propose a minimal safe change."))

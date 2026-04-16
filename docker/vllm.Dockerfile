FROM vllm/vllm-openai:latest
ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# Entrypoint configured via docker-compose

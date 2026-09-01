FROM vllm/vllm-openai:glm53-flash
RUN echo "dflash2-overlay-sm100-20260901"
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY overlay/qwen3_dflash2.py        $VLLM/model_executor/models/qwen3_dflash2.py
COPY overlay/dflash2/                $VLLM/v1/worker/gpu/spec_decode/dflash2/
COPY overlay/patch_registry_and_select.py /tmp/patch_registry_and_select.py
COPY overlay/patch_glm_aux_capture.py       /tmp/patch_glm_aux_capture.py
COPY overlay/patch_kv_page_lcm2.py          /tmp/patch_kv_page_lcm2.py
COPY overlay/patch_glm5_drafter_group.py    /tmp/patch_glm5_drafter_group.py
COPY overlay/sim_glm5_drafter_hades.py /tmp/sim_glm5_drafter_hades.py
RUN python3 /tmp/patch_registry_and_select.py \
 && python3 /tmp/patch_glm_aux_capture.py \
 && python3 /tmp/patch_kv_page_lcm2.py \
 && python3 /tmp/patch_glm5_drafter_group.py \
 && python3 -c "from vllm.model_executor.models.registry import ModelRegistry; assert \"DFlash2DraftModel\" in ModelRegistry.get_supported_archs(); print('DFlash2 overlay build check OK')" \
 && python3 /tmp/sim_glm5_drafter_hades.py \
 && echo 'DFlash2 SM100/GB300 geometry validation OK'

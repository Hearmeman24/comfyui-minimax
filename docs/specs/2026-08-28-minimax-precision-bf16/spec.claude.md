# Choose full bf16 MiniMax H3 diffusion models at deploy time

- **Work type:** `feature/app`
- **Status:** `draft` → proceed under task-scoped authority; no material decision is unresolved
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context

The template currently exposes `minimax_quant` with int8, fp8, and nvfp4 profiles, and defaults to int8. It has no deploy-time route to Comfy-Org's full bf16 FL2VA and Ref2VA files. — evidence: `template.json:18-42`

## 2. Approach & Why

- Add both bf16 filenames to the registry, because every selected swap-profile filename must resolve through the registry before it can be queued. — evidence: `src/models_registry.json:22-40`, `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:148-153`
- Add bf16 to the existing `minimax_quant` swap group internally, then let the template-local pre-download hook translate `minimax_precision=bf16` to that internal profile. The hook is sourced before the provisioner and may export environment changes the provisioner reads. — evidence: `src/hooks/pre_download.sh:1-8`, `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/start.sh:605-636`
- Leave unset/`convrot` precision unchanged so the existing int8/fp8/nvfp4 selection remains authoritative. The current default and profile files are defined in one swap group. — evidence: `template.json:18-42`
- Use the existing unknown-value behavior: normalize case and whitespace, warn, and retain the default/current selection instead of aborting boot. — evidence: `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:103-118`

## 3. Acceptance Criteria

- [ ] Leaving `minimax_precision` unset, or setting it to `convrot`, keeps today's `minimax_quant`-selected diffusion files and workflows. → (ask: "it's convrot by default and installed the models that are currently on the template")
- [ ] Setting `minimax_precision=bf16` queues `minimax_h3_fl2va_bf16.safetensors` and `minimax_h3_ref2va_bf16.safetensors` instead of either quantized diffusion pair. → (ask: "if it's bf16 install ... Instead")
- [ ] The accepted public precision values are unset/default-convrot, `convrot`, and `bf16`, with case and surrounding whitespace normalized consistently with current profile handling. → (ask: "accepts either convrot or bf16")
- [ ] Copied workflows reference the selected diffusion files in both loader widgets and `properties.models`, matching the queued manifest. → (ask: "install ... Instead")

## 4. Scope & Non-Goals

**In scope:** `template.json:18-42`, `src/models_registry.json:22-40`, `src/hooks/pre_download.sh:1-28`, `tools/test_provisioner.py:1-210`, `.circleci/config.yml:42-90`, `README.md:13-50`, and the in-ComfyUI settings guide at `src/note_sections.md:85-110`.

**Non-goals (explicitly NOT doing):** changing the shared runtime; changing VAE, text-encoder, or bundled LoRA selection; modifying source workflow JSON; deploying, publishing, or downloading model payloads locally.

## 5. Key Decisions & Constraints

- **Decided:** bf16 precision overrides `minimax_quant`, while unset/convrot preserves the current quant selector. This is the only way "instead" avoids queuing both profiles while retaining the existing public quant contract. — evidence: `template.json:18-42`, `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:121-158`
- **Constraint / must-not-break:** only the active swap profile is queued, and workflow loader values plus Missing Models URLs must agree with it. — evidence: `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:161-189`, `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:382-408`
- **Mirror existing:** `src/hooks/pre_download.sh:1-8` for sourced, idempotent, non-fatal hook behavior; `tools/test_provisioner.py:145-205` for manifest/workflow agreement and fallback regression coverage.
- **Scale:** Each bf16 diffusion file is 66,280,487,368 bytes upstream, so the volume guidance must account for roughly 132.6 GB for this pair alone.

## 6. Code Surface Map

- `template.json:18-42` — authoritative swap-profile map.
- `src/models_registry.json:22-40` — authoritative URL and destination metadata for diffusion models.
- `src/hooks/pre_download.sh:1-28` — template-local environment translation before provisioning.
- `tools/test_provisioner.py:102-205` — highest behavioral seam for selected manifest and copied workflow agreement.
- `README.md:13-50` — deploy-time environment and volume guidance.
- `src/note_sections.md:85-110` — settings guidance installed into the user's ComfyUI workflow list.
- `.circleci/config.yml:42-90` — repository verification gate and test label.

## 7. Ultracode Dispatch Notes

**Build first (sequential — freezes interfaces before any parallelism):**
- Define the profile precedence in `template.json` and `src/hooks/pre_download.sh` together.

**Parallel slices:**
- None. The registry, profile, hook, regression, and documentation form one small vertical slice with overlapping contract ownership.

**⛓ Collision audit:** A single serialized implementation owns all writes; no slice collision exists.

**The implementer must:** add the feature, green the behavioral test, run live URL validation, and verify the final diff against §3.

```yaml
dispatch:
  frozen:
    - workflows/
    - pins.json
  slices:
    - {key: precisionProfile, writes: [template.json, src/models_registry.json, src/hooks/pre_download.sh, tools/test_provisioner.py, .circleci/config.yml, README.md, src/note_sections.md]}
  testRunner: "python3 tools/test_provisioner.py"
```

## 8. Assumptions & Open Questions

None. Both requested upstream filenames were confirmed in the Comfy-Org/MiniMax-H3 repository, the current hook order was inspected, and the existing fallback/profile behavior supplies the remaining contract.

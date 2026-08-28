# Use minimax_quant=false for full bf16 models

- **Work type:** `refactor/migration`
- **Status:** `draft` → proceed under task-scoped authority; no material decision is unresolved
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context

The previous local commit introduced `minimax_precision` and translated it to an internal bf16 `minimax_quant` profile in the pre-download hook. The user has superseded that interface: the existing `minimax_quant` environment variable itself should accept `false`. — evidence: `src/hooks/pre_download.sh:10-32`, `template.json:18-47`

## 2. Approach & Why

- Rename the swap profile key from `bf16` to the string `false`. The shared provisioner reads environment values as strings, matches exact profile keys, then normalized lowercase keys; therefore `false`, `FALSE`, and surrounding whitespace resolve without a template-specific hook. — evidence: `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:103-118`, `template.json:18-47`
- Remove only the `minimax_precision` translation block from the pre-download hook; retain the adjacent OpenRouter cleanup unchanged. — evidence: `src/hooks/pre_download.sh:1-52`
- Keep the existing bf16 registry entries and profile filenames because they already point to the requested upstream files and passed live validation. — evidence: `src/models_registry.json:22-48`, `template.json:41-45`

## 3. Acceptance Criteria

- [ ] `minimax_quant=false` queues `minimax_h3_fl2va_bf16.safetensors` and `minimax_h3_ref2va_bf16.safetensors`, and no int8/fp8 diffusion file. → (ask: "set minimax_quant to false and it will download the bf16 models")
- [ ] Unset `minimax_quant` remains int8; int8, fp8, and nvfp4 remain unchanged. → (ask: "give an option")
- [ ] Case and surrounding whitespace are normalized, and an unknown value still warns and falls back to int8. → (ask: "minimax_quant to false")
- [ ] `minimax_precision` no longer appears in runtime code, tests, README, or the installed ComfyUI settings note. → (ask: "So let's give an option to set minimax_quant")

## 4. Scope & Non-Goals

**In scope:** `template.json:18-47`, `src/hooks/pre_download.sh:1-52`, `tools/test_provisioner.py:1-273`, `README.md:16-58`, `src/note_sections.md:85-124`, and `.circleci/config.yml:42-90`.

**Non-goals:** changing model URLs, VAE/text-encoder/LoRA selection, source workflows, the shared runtime, deployment, publishing, or downloading model payloads locally.

## 5. Key Decisions & Constraints

- **Invariant:** `minimax_quant` stays default `int8`; only an explicit false profile selects bf16. — evidence: `template.json:20-45`
- **Invariant:** selected workflow loader names and `properties.models` URLs must match the manifest, with exactly one managed diffusion pair. — evidence: `tools/test_provisioner.py:223-240`
- **Mirror existing:** the provisioner's current case/whitespace normalization and unknown-value fallback. — evidence: `/Users/avivkaplan/.cache/comfyui-runtime-validator/src/provisioner.py:103-118`
- **Scale:** The two bf16 diffusion files remain 66,280,487,368 bytes each, so operator guidance remains at least 180 GB and preferably 200 GB.

## 6. Code Surface Map

- `template.json:18-47` — authoritative public environment-to-profile map.
- `src/hooks/pre_download.sh:1-52` — remove superseded precision translation; preserve unrelated cleanup.
- `tools/test_provisioner.py:55-268` — behavioral profile, fallback, workflow, and manifest regression.
- `README.md:16-58` — deploy-time variable and storage guidance.
- `src/note_sections.md:85-124` — settings guidance installed into ComfyUI.
- `.circleci/config.yml:42-90` — CI gate description.

## 7. Ultracode Dispatch Notes

**Build first:** Rename the profile and remove the translation as one interface change.

**Parallel slices:** None; the test and both guidance surfaces describe that same small interface.

**⛓ Collision audit:** One serialized implementation owns all writes.

```yaml
dispatch:
  frozen:
    - src/models_registry.json
    - workflows/
    - pins.json
  slices:
    - {key: quantFalse, writes: [template.json, src/hooks/pre_download.sh, tools/test_provisioner.py, README.md, src/note_sections.md, .circleci/config.yml]}
  testRunner: "python3 tools/test_provisioner.py"
```

## 8. Assumptions & Open Questions

None. The requested interface is explicit, and the pinned provisioner's exact string-profile resolution was inspected.

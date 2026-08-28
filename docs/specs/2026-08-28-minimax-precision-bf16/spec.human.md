# Choose full bf16 MiniMax H3 diffusion models at deploy time

**Type:** `feature/app` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done

Leaving `minimax_precision` unset, or setting it to `convrot`, keeps today's model selection. Setting it to `bf16` downloads the full bf16 FL2VA and Ref2VA diffusion models instead and rewrites every installed workflow to use them.

## 🪤 Gotchas

- `bf16` must override `minimax_quant`; otherwise the provisioner would queue both the quantized and bf16 diffusion models.
- The two bf16 diffusion files are about 66.3 GB each, so bf16 deployments need at least 180 GB of volume space; 200 GB is the comfortable recommendation.
- Unknown precision values follow the existing safe boot convention: warn and retain the current quant/default model selection.

## Done when

- [ ] Unset/`convrot` precision installs the same diffusion models selected today by `minimax_quant`.
- [ ] `minimax_precision=bf16` installs only `minimax_h3_fl2va_bf16.safetensors` and `minimax_h3_ref2va_bf16.safetensors` in place of the quantized diffusion pair.
- [ ] Installed workflows name the selected pair in loader widgets and Missing Models metadata.
- [ ] The environment-variable and storage requirements are documented.

## The plan

1. Register the two upstream bf16 files and add a bf16 provisioning profile.
2. Translate `minimax_precision=bf16` into that profile in the existing pre-download hook.
3. Expand the provisioning regression across unset, convrot, bf16, mixed quant/precision, case-normalized, and invalid values.
4. Run the repository's static, provisioning, and live model validation gates.

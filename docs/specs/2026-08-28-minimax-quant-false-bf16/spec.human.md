# Use minimax_quant=false for full bf16 models

**Type:** `refactor/migration` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done

There is no `minimax_precision` setting. The existing `minimax_quant` setting accepts `false`; setting it to `false` downloads the full bf16 FL2VA and Ref2VA diffusion models, while unset still defaults to int8.

## 🪤 Gotchas

- `false` is a profile value for `minimax_quant`, not the model-download flag. `download_minimax_h3` must still be `true`.
- Existing int8, fp8, and nvfp4 behavior must stay byte-for-byte equivalent in manifest terms.
- The bf16 pair is about 132.6 GB before the unchanged encoder, VAEs, and LoRAs, so the 180/200 GB storage guidance remains necessary.

## Done when

- [ ] `minimax_quant=false` installs only the requested full bf16 diffusion pair instead of a quantized pair.
- [ ] Unset, int8, fp8, nvfp4, case normalization, and invalid fallback continue working.
- [ ] `minimax_precision` is removed from runtime behavior, tests, README, and the installed ComfyUI note.
- [ ] Workflows and Missing Models metadata agree with the selected manifest.

## The plan

1. Rename the internal bf16 swap profile to the public `false` value.
2. Remove the now-unneeded precision translation from the pre-download hook.
3. Collapse the regression suite back to one environment variable and cover false/FALSE/whitespace.
4. Update both user-guidance surfaces and run static plus live model validation.

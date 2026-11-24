# Incremental Speech Separation (ISS)

This repository contains a runnable source implementation of the ISS speech-separation and speech-denoising workflows described in *Learning to Separate Continually: A Unified Framework for Incremental Speech Separation*. It includes the waveform model, task manifests and loading, source-domain initialization, sequential CSA/DSA training, DPAN, evaluation, and single-file inference.


## Method And Scope

The model is a waveform SepFormer. It uses a 256-channel convolutional encoder, a dual-path mask network with two dual-path blocks, 250-frame chunks, eight intra and eight inter Transformer layers per block, and a transposed-convolution decoder. The formal attention layer is batch-first scaled dot-product attention with one residual connection per attention/FFN sublayer. A SaRA linear layer adds shared CSA and current-domain DSA low-rank updates to Q/K/V/output projections and both FFN projections. The task configs set total rank four: separation uses shared/specific ranks `[2, 2]`; denoising uses `[1, 3]`. Adapter scaling is 0.25. The 24 earliest Transformer layers use activation checkpointing.

The source domain is trained separately. For a sequential run, the public config order is source domain followed by the four incremental domains:

- Separation: English, Yoruba, Tamil, Catalan, Mandarin.
- Denoising: base (the 10-noise source model), Alarm, Cough, Destroyerops/DD, MachineGun/MG. The machine-readable manifest IDs are `base`, `alarm`, `cough`, `dd`, and `mg`.

The DPAN fitted by this protocol predicts among the four incremental domains only. Its stage-4 class map is `[Yoruba, Tamil, Catalan, Mandarin]` for separation and `[alarm, cough, dd, mg]` for denoising. It does not contain a source-domain/base class. Consequently, the included DPAN-routed inference command is for those four incremental domains; source-domain routing needs a separately specified and trained selector protocol.

The incremental protocol uses Adam at `1.5e-4`, gradient clipping at 5, full waveforms, batch size 4, 20 epochs per incremental domain, and FP16 autocast on CUDA. It applies 95/100/105% speed perturbation. Denoising training re-adds the unperturbed noise target after speed perturbing the clean speech. The loss is negative SpeechBrain PIT-SI-SNR. From stage 2 onward, the shared adapter drift regularizer is `1e-3 * ||CSA - inherited_CSA||^2 / (||inherited_CSA||^2 + 5e-4)`. Validation chooses the best epoch per domain: SI-SNR for separation and PESQ narrowband for denoising. DPAN uses frozen source-encoder features, full-time temporal mean pooling, empirical multivariate covariance, Gaussian-scale-mixture multivariate Laplace pseudo-features, and a linear classifier trained with cross-entropy, SGD (`lr=10`, `momentum=0.9`), cosine annealing, 1000 epochs, and batch size 128. Source-domain initialization uses the separate budgets in `training.source_epochs` (55 separation, 200 denoising) and selects with `training.selection` (SI-SNR for separation, narrowband PESQ for denoising), reducing its learning rate on validation plateaus.

## Layout

- `iss/models/`: dual-path network, the formal batch-first Transformer layer, ISS Adapter, full task model construction, and base checkpoint conversion.
- `iss/data/`: audio validation, manifest conversion, dataset, and waveform collation.
- `iss/training/`: source initialization, sequential adapter training, PIT-SI-SNR, optimizer state, freezing, and stage checkpoint management.
- `iss/routing/`: encoder feature extraction, DPAN distributions, pseudo-feature replay, selector training, and domain prediction.
- `iss/evaluation/`: stage and domain validation, SI-SNR/SDR/PESQ, DPAN/oracle evaluation, and single-file inference.
- `configs/`: complete separation and denoising YAML configs. These are plain YAML dictionaries, not HyperPyYAML object graphs; constructors are explicit in `iss.models.network`.
- `scripts/`: data manifest, source/incremental/DPAN training, evaluation, inference, config check, and synthetic smoke commands.
- `THIRD_PARTY_LICENSES/`: SpeechBrain license text. See `THIRD_PARTY_SOURCES.md` and `LICENSE-STATUS.md`.

## Install

Python 3.10 or newer, PyTorch 2.3 or newer, and a CUDA-capable PyTorch build are recommended for full training. On CPU, use `device: cpu` in the selected config.

```bash
python -m pip install -e .
python scripts/check_configs.py
python scripts/smoke_check.py
```


Example layout and commands for one separation domain:

```bash
python scripts/generate_separation_mixtures.py --pairs-csv /data/splits/English_tr_pairs.csv --split tr --output-root /data/language_mix/English/2speakers
python scripts/generate_separation_mixtures.py --pairs-csv /data/splits/English_cv_pairs.csv --split cv --output-root /data/language_mix/English/2speakers
python scripts/generate_separation_mixtures.py --pairs-csv /data/splits/English_tt_pairs.csv --split tt --output-root /data/language_mix/English/2speakers
python scripts/prepare_wsjmix.py --data-root /data/language_mix/English/2speakers --output-dir /data/prepared/English
```

For denoising, use preassigned clean/noise/SNR rows for each split:

```bash
python scripts/generate_noisy_mixtures.py --items-csv /data/splits/alarm_tr.csv --split tr --output-root /data/wsj0/generated/alarm --seed 1234
python scripts/generate_noisy_mixtures.py --items-csv /data/splits/alarm_cv.csv --split cv --output-root /data/wsj0/generated/alarm --seed 1234
python scripts/generate_noisy_mixtures.py --items-csv /data/splits/alarm_tt.csv --split tt --output-root /data/wsj0/generated/alarm --seed 1234
python scripts/prepare_enhancement_csv.py --data-root /data/wsj0/generated/alarm --output-dir /data/prepared/alarm
```

Repeat with `Yoruba`, `Tamil`, `Catalan`, and `Mandarin` pair lists, and with `cough`, `dd`, `mg`, and any verified base denoising data. 

`prepare_manifests.py` preserves the CSV row order and split. For each CSV, required fields are `ID`, `mix_wav`, and `s1_wav`; separation also needs `s2_wav`, and denoising may include `noise_wav`. Relative paths are resolved under `--audio-root`; absolute paths in user-supplied CSVs remain absolute in the generated local manifest.

For corpus trees already arranged like the public recipes, the metadata builders create recipe CSVs without changing the audio or split:

```bash
python scripts/prepare_wsjmix.py --data-root /data/wsj0/2speakers --output-dir /data/prepared/English
python scripts/prepare_enhancement_csv.py --data-root /data/wsj0/generated/alarm --output-dir /data/prepared/alarm
```

The first writes `wsj_tr.csv`, `wsj_cv.csv`, and `wsj_tt.csv` from `wav8k/min/{tr,cv,tt}/{mix,s1,s2}`. The second writes `wsj0_*.csv` from `wav8k/min/{tr,cv,tt}/{mix_single,s1,noise}`. These metadata helpers index already prepared mixtures and targets. The synthesis scripts above create audio from explicitly enumerated row lists; they do not create the original fixed split or recover its row assignments.

Example for one already-prepared split:

```bash
python scripts/prepare_manifests.py --task separation --domain English --split tr --csv /data/recipe/English_tr.csv --audio-root /data/audio --output data/manifests/separation/English_tr.jsonl --sample-rate 8000
```

Repeat the same command for `cv` and `tt`, then for Yoruba, Tamil, Catalan, and Mandarin. For denoising, use `--task denoising`, domains `base`, `alarm`, `cough`, `dd`, and `mg`, and splits `tr`, `cv`, and `tt`. The display aliases are Base10, Alarm, Cough, DD/Destroyerops, and MG/MachineGun. Each JSONL row records an utterance ID, task/domain/split, sample rate, mixture path, and target paths. At runtime all referenced audio must be finite mono 8-kHz WAV; separation needs two target sources, and denoising training needs the clean source and noise target.


## Train And Use

Run commands from the repository root. The source command trains a task-specific source-domain initializer and selects it with the task's configured validation metric. Its output path is `source_output_dir/base_model.pt`; set `base_checkpoint` to that file (the supplied configs already point there). The paper-era stored base states are not included, and a newly trained initializer is not bit-identical to them.

```bash
python scripts/train_source.py --config configs/separation.yaml
python scripts/train_incremental.py --config configs/separation.yaml
python scripts/train_dpan.py --config configs/separation.yaml
python scripts/evaluate.py --config configs/separation.yaml
python scripts/infer.py --config configs/separation.yaml --input /path/to/mixture.wav --output /path/to/estimate.wav
```

The trajectory directory, DPAN directory, feature output, and general run directory are configurable at the top level of each YAML. Training writes validation-selected `stageN_DOMAIN/best.pt` states and a resumable `last.pt`; DPAN writes one compressed distribution and selector per stage. Evaluation runs DPAN routing and a true-domain oracle reference over the four incremental test domains. Oracle labels are used only for this evaluation reference, never for inference. SI-SNR uses SpeechBrain `get_si_snr_with_pitwrapper`; SDR uses `mir_eval.bss_eval_sources`; PESQ uses 8-kHz narrowband mode. Per-utterance errors and metric coverage are retained in the evaluation output.


## Citation

Please cite the manuscript as *Learning to Separate Continually: A Unified Framework for Incremental Speech Separation*, together with SpeechBrain 0.5.16 and the source corpora used. 
Once the manuscript is accepted, the references will be updated., 